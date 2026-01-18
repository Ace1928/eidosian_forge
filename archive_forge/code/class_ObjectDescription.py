import re
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives, roles
from sphinx import addnodes
from sphinx.addnodes import desc_signature
from sphinx.util import docutils
from sphinx.util.docfields import DocFieldTransformer, Field, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import OptionSpec
class ObjectDescription(SphinxDirective, Generic[T]):
    """
    Directive to describe a class, function or similar object.  Not used
    directly, but subclassed (in domain-specific directives) to add custom
    behavior.
    """
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec: OptionSpec = {'noindex': directives.flag, 'noindexentry': directives.flag, 'nocontentsentry': directives.flag}
    doc_field_types: List[Field] = []
    domain: Optional[str] = None
    objtype: Optional[str] = None
    indexnode: Optional[addnodes.index] = None
    _doc_field_type_map: Dict[str, Tuple[Field, bool]] = {}

    def get_field_type_map(self) -> Dict[str, Tuple[Field, bool]]:
        if self._doc_field_type_map == {}:
            self._doc_field_type_map = {}
            for field in self.doc_field_types:
                for name in field.names:
                    self._doc_field_type_map[name] = (field, False)
                if field.is_typed:
                    typed_field = cast(TypedField, field)
                    for name in typed_field.typenames:
                        self._doc_field_type_map[name] = (field, True)
        return self._doc_field_type_map

    def get_signatures(self) -> List[str]:
        """
        Retrieve the signatures to document from the directive arguments.  By
        default, signatures are given as arguments, one per line.
        """
        lines = nl_escape_re.sub('', self.arguments[0]).split('\n')
        if self.config.strip_signature_backslash:
            return [strip_backslash_re.sub('\\1', line.strip()) for line in lines]
        else:
            return [line.strip() for line in lines]

    def handle_signature(self, sig: str, signode: desc_signature) -> T:
        """
        Parse the signature *sig* into individual nodes and append them to
        *signode*. If ValueError is raised, parsing is aborted and the whole
        *sig* is put into a single desc_name node.

        The return value should be a value that identifies the object.  It is
        passed to :meth:`add_target_and_index()` unchanged, and otherwise only
        used to skip duplicates.
        """
        raise ValueError

    def add_target_and_index(self, name: T, sig: str, signode: desc_signature) -> None:
        """
        Add cross-reference IDs and entries to self.indexnode, if applicable.

        *name* is whatever :meth:`handle_signature()` returned.
        """
        return

    def before_content(self) -> None:
        """
        Called before parsing content. Used to set information about the current
        directive context on the build environment.
        """
        pass

    def transform_content(self, contentnode: addnodes.desc_content) -> None:
        """
        Called after creating the content through nested parsing,
        but before the ``object-description-transform`` event is emitted,
        and before the info-fields are transformed.
        Can be used to manipulate the content.
        """
        pass

    def after_content(self) -> None:
        """
        Called after parsing content. Used to reset information about the
        current directive context on the build environment.
        """
        pass

    def _object_hierarchy_parts(self, sig_node: desc_signature) -> Tuple[str, ...]:
        """
        Returns a tuple of strings, one entry for each part of the object's
        hierarchy (e.g. ``('module', 'submodule', 'Class', 'method')``). The
        returned tuple is used to properly nest children within parents in the
        table of contents, and can also be used within the
        :py:meth:`_toc_entry_name` method.

        This method must not be used outwith table of contents generation.
        """
        return ()

    def _toc_entry_name(self, sig_node: desc_signature) -> str:
        """
        Returns the text of the table of contents entry for the object.

        This function is called once, in :py:meth:`run`, to set the name for the
        table of contents entry (a special attribute ``_toc_name`` is set on the
        object node, later used in
        ``environment.collectors.toctree.TocTreeCollector.process_doc().build_toc()``
        when the table of contents entries are collected).

        To support table of contents entries for their objects, domains must
        override this method, also respecting the configuration setting
        ``toc_object_entries_show_parents``. Domains must also override
        :py:meth:`_object_hierarchy_parts`, with one (string) entry for each part of the
        object's hierarchy. The result of this method is set on the signature
        node, and can be accessed as ``sig_node['_toc_parts']`` for use within
        this method. The resulting tuple is also used to properly nest children
        within parents in the table of contents.

        An example implementations of this method is within the python domain
        (:meth:`PyObject._toc_entry_name`). The python domain sets the
        ``_toc_parts`` attribute within the :py:meth:`handle_signature()`
        method.
        """
        return ''

    def run(self) -> List[Node]:
        """
        Main directive entry function, called by docutils upon encountering the
        directive.

        This directive is meant to be quite easily subclassable, so it delegates
        to several additional methods.  What it does:

        * find out if called as a domain-specific directive, set self.domain
        * create a `desc` node to fit all description inside
        * parse standard options, currently `noindex`
        * create an index node if needed as self.indexnode
        * parse all given signatures (as returned by self.get_signatures())
          using self.handle_signature(), which should either return a name
          or raise ValueError
        * add index entries using self.add_target_and_index()
        * parse the content and handle doc fields in it
        """
        if ':' in self.name:
            self.domain, self.objtype = self.name.split(':', 1)
        else:
            self.domain, self.objtype = ('', self.name)
        self.indexnode = addnodes.index(entries=[])
        node = addnodes.desc()
        node.document = self.state.document
        source, line = self.get_source_info()
        if line is not None:
            line -= 1
        self.state.document.note_source(source, line)
        node['domain'] = self.domain
        node['objtype'] = node['desctype'] = self.objtype
        node['noindex'] = noindex = 'noindex' in self.options
        node['noindexentry'] = 'noindexentry' in self.options
        node['nocontentsentry'] = 'nocontentsentry' in self.options
        if self.domain:
            node['classes'].append(self.domain)
        node['classes'].append(node['objtype'])
        self.names: List[T] = []
        signatures = self.get_signatures()
        for sig in signatures:
            signode = addnodes.desc_signature(sig, '')
            self.set_source_info(signode)
            node.append(signode)
            try:
                name = self.handle_signature(sig, signode)
            except ValueError:
                signode.clear()
                signode += addnodes.desc_name(sig, sig)
                continue
            finally:
                if self.env.app.config.toc_object_entries:
                    signode['_toc_parts'] = self._object_hierarchy_parts(signode)
                    signode['_toc_name'] = self._toc_entry_name(signode)
                else:
                    signode['_toc_parts'] = ()
                    signode['_toc_name'] = ''
            if name not in self.names:
                self.names.append(name)
                if not noindex:
                    self.add_target_and_index(name, sig, signode)
        contentnode = addnodes.desc_content()
        node.append(contentnode)
        if self.names:
            self.env.temp_data['object'] = self.names[0]
        self.before_content()
        nested_parse_with_titles(self.state, self.content, contentnode)
        self.transform_content(contentnode)
        self.env.app.emit('object-description-transform', self.domain, self.objtype, contentnode)
        DocFieldTransformer(self).transform_all(contentnode)
        self.env.temp_data['object'] = None
        self.after_content()
        return [self.indexnode, node]