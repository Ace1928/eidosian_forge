from __future__ import (absolute_import, division, print_function)
from collections.abc import MutableMapping, MutableSet, MutableSequence
from pathlib import Path
from ansible import constants as C
from ansible.release import __version__ as ansible_version
from ansible.errors import AnsibleError, AnsibleParserError, AnsiblePluginNotFound
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.plugin_docs import read_docstring
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def add_fragments(doc, filename, fragment_loader, is_module=False):
    fragments = doc.pop('extends_documentation_fragment', [])
    if isinstance(fragments, string_types):
        fragments = [fragments]
    unknown_fragments = []
    for fragment_slug in fragments:
        fragment_name = fragment_slug
        fragment_var = 'DOCUMENTATION'
        fragment_class = fragment_loader.get(fragment_name)
        if fragment_class is None and '.' in fragment_slug:
            splitname = fragment_slug.rsplit('.', 1)
            fragment_name = splitname[0]
            fragment_var = splitname[1].upper()
            fragment_class = fragment_loader.get(fragment_name)
        if fragment_class is None:
            unknown_fragments.append(fragment_slug)
            continue
        fragment_yaml = getattr(fragment_class, fragment_var, None)
        if fragment_yaml is None:
            if fragment_var != 'DOCUMENTATION':
                unknown_fragments.append(fragment_slug)
                continue
            else:
                fragment_yaml = '{}'
        fragment = AnsibleLoader(fragment_yaml, file_name=filename).get_single_data()
        real_fragment_name = getattr(fragment_class, 'ansible_name')
        real_collection_name = '.'.join(real_fragment_name.split('.')[0:2]) if '.' in real_fragment_name else ''
        add_collection_to_versions_and_dates(fragment, real_collection_name, is_module=is_module)
        if 'notes' in fragment:
            notes = fragment.pop('notes')
            if notes:
                if 'notes' not in doc:
                    doc['notes'] = []
                doc['notes'].extend(notes)
        if 'seealso' in fragment:
            seealso = fragment.pop('seealso')
            if seealso:
                if 'seealso' not in doc:
                    doc['seealso'] = []
                doc['seealso'].extend(seealso)
        if 'options' not in fragment and 'attributes' not in fragment:
            raise Exception('missing options or attributes in fragment (%s), possibly misformatted?: %s' % (fragment_name, filename))
        for doc_key in ['options', 'attributes']:
            if doc_key in fragment:
                if doc_key in doc:
                    try:
                        merge_fragment(doc[doc_key], fragment.pop(doc_key))
                    except Exception as e:
                        raise AnsibleError('%s %s (%s) of unknown type: %s' % (to_native(e), doc_key, fragment_name, filename))
                else:
                    doc[doc_key] = fragment.pop(doc_key)
        try:
            merge_fragment(doc, fragment)
        except Exception as e:
            raise AnsibleError('%s (%s) of unknown type: %s' % (to_native(e), fragment_name, filename))
    if unknown_fragments:
        raise AnsibleError('unknown doc_fragment(s) in file {0}: {1}'.format(filename, to_native(', '.join(unknown_fragments))))