import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class Substitutions(Transform):
    """
    Given the following ``document`` as input::

        <document>
            <paragraph>
                The
                <substitution_reference refname="biohazard">
                    biohazard
                 symbol is deservedly scary-looking.
            <substitution_definition name="biohazard">
                <image alt="biohazard" uri="biohazard.png">

    The ``substitution_reference`` will simply be replaced by the
    contents of the corresponding ``substitution_definition``.

    The transformed result will be::

        <document>
            <paragraph>
                The
                <image alt="biohazard" uri="biohazard.png">
                 symbol is deservedly scary-looking.
            <substitution_definition name="biohazard">
                <image alt="biohazard" uri="biohazard.png">
    """
    default_priority = 220
    'The Substitutions transform has to be applied very early, before\n    `docutils.tranforms.frontmatter.DocTitle` and others.'

    def apply(self):
        defs = self.document.substitution_defs
        normed = self.document.substitution_names
        subreflist = self.document.traverse(nodes.substitution_reference)
        nested = {}
        for ref in subreflist:
            refname = ref['refname']
            key = None
            if refname in defs:
                key = refname
            else:
                normed_name = refname.lower()
                if normed_name in normed:
                    key = normed[normed_name]
            if key is None:
                msg = self.document.reporter.error('Undefined substitution referenced: "%s".' % refname, base_node=ref)
                msgid = self.document.set_id(msg)
                prb = nodes.problematic(ref.rawsource, ref.rawsource, refid=msgid)
                prbid = self.document.set_id(prb)
                msg.add_backref(prbid)
                ref.replace_self(prb)
            else:
                subdef = defs[key]
                parent = ref.parent
                index = parent.index(ref)
                if 'ltrim' in subdef.attributes or 'trim' in subdef.attributes:
                    if index > 0 and isinstance(parent[index - 1], nodes.Text):
                        parent[index - 1] = parent[index - 1].rstrip()
                if 'rtrim' in subdef.attributes or 'trim' in subdef.attributes:
                    if len(parent) > index + 1 and isinstance(parent[index + 1], nodes.Text):
                        parent[index + 1] = parent[index + 1].lstrip()
                subdef_copy = subdef.deepcopy()
                try:
                    for nested_ref in subdef_copy.traverse(nodes.substitution_reference):
                        nested_name = normed[nested_ref['refname'].lower()]
                        if nested_name in nested.setdefault(nested_name, []):
                            raise CircularSubstitutionDefinitionError
                        else:
                            nested[nested_name].append(key)
                            nested_ref['ref-origin'] = ref
                            subreflist.append(nested_ref)
                except CircularSubstitutionDefinitionError:
                    parent = ref.parent
                    if isinstance(parent, nodes.substitution_definition):
                        msg = self.document.reporter.error('Circular substitution definition detected:', nodes.literal_block(parent.rawsource, parent.rawsource), line=parent.line, base_node=parent)
                        parent.replace_self(msg)
                    else:
                        ref_origin = ref
                        while ref_origin.hasattr('ref-origin'):
                            ref_origin = ref_origin['ref-origin']
                        msg = self.document.reporter.error('Circular substitution definition referenced: "%s".' % refname, base_node=ref_origin)
                        msgid = self.document.set_id(msg)
                        prb = nodes.problematic(ref.rawsource, ref.rawsource, refid=msgid)
                        prbid = self.document.set_id(prb)
                        msg.add_backref(prbid)
                        ref.replace_self(prb)
                else:
                    ref.replace_self(subdef_copy.children)
                    for node in subdef_copy.children:
                        if isinstance(node, nodes.Referential):
                            if 'refname' in node:
                                self.document.note_refname(node)