import sys
import os
import re
import warnings
import types
import unicodedata
class document(Root, Structural, Element):
    """
    The document root element.

    Do not instantiate this class directly; use
    `docutils.utils.new_document()` instead.
    """

    def __init__(self, settings, reporter, *args, **kwargs):
        Element.__init__(self, *args, **kwargs)
        self.current_source = None
        'Path to or description of the input source being processed.'
        self.current_line = None
        'Line number (1-based) of `current_source`.'
        self.settings = settings
        'Runtime settings data record.'
        self.reporter = reporter
        'System message generator.'
        self.indirect_targets = []
        'List of indirect target nodes.'
        self.substitution_defs = {}
        'Mapping of substitution names to substitution_definition nodes.'
        self.substitution_names = {}
        'Mapping of case-normalized substitution names to case-sensitive\n        names.'
        self.refnames = {}
        'Mapping of names to lists of referencing nodes.'
        self.refids = {}
        'Mapping of ids to lists of referencing nodes.'
        self.nameids = {}
        "Mapping of names to unique id's."
        self.nametypes = {}
        'Mapping of names to hyperlink type (boolean: True => explicit,\n        False => implicit.'
        self.ids = {}
        'Mapping of ids to nodes.'
        self.footnote_refs = {}
        'Mapping of footnote labels to lists of footnote_reference nodes.'
        self.citation_refs = {}
        'Mapping of citation labels to lists of citation_reference nodes.'
        self.autofootnotes = []
        'List of auto-numbered footnote nodes.'
        self.autofootnote_refs = []
        'List of auto-numbered footnote_reference nodes.'
        self.symbol_footnotes = []
        'List of symbol footnote nodes.'
        self.symbol_footnote_refs = []
        'List of symbol footnote_reference nodes.'
        self.footnotes = []
        'List of manually-numbered footnote nodes.'
        self.citations = []
        'List of citation nodes.'
        self.autofootnote_start = 1
        'Initial auto-numbered footnote number.'
        self.symbol_footnote_start = 0
        'Initial symbol footnote symbol index.'
        self.id_start = 1
        'Initial ID number.'
        self.parse_messages = []
        'System messages generated while parsing.'
        self.transform_messages = []
        'System messages generated while applying transforms.'
        import docutils.transforms
        self.transformer = docutils.transforms.Transformer(self)
        'Storage for transforms to be applied to this document.'
        self.decoration = None
        "Document's `decoration` node."
        self.document = self

    def __getstate__(self):
        """
        Return dict with unpicklable references removed.
        """
        state = self.__dict__.copy()
        state['reporter'] = None
        state['transformer'] = None
        return state

    def asdom(self, dom=None):
        """Return a DOM representation of this document."""
        if dom is None:
            import xml.dom.minidom as dom
        domroot = dom.Document()
        domroot.appendChild(self._dom_node(domroot))
        return domroot

    def set_id(self, node, msgnode=None):
        for id in node['ids']:
            if id in self.ids and self.ids[id] is not node:
                msg = self.reporter.severe('Duplicate ID: "%s".' % id)
                if msgnode != None:
                    msgnode += msg
        if not node['ids']:
            for name in node['names']:
                id = self.settings.id_prefix + make_id(name)
                if id and id not in self.ids:
                    break
            else:
                id = ''
                while not id or id in self.ids:
                    id = self.settings.id_prefix + self.settings.auto_id_prefix + str(self.id_start)
                    self.id_start += 1
            node['ids'].append(id)
        self.ids[id] = node
        return id

    def set_name_id_map(self, node, id, msgnode=None, explicit=None):
        """
        `self.nameids` maps names to IDs, while `self.nametypes` maps names to
        booleans representing hyperlink type (True==explicit,
        False==implicit).  This method updates the mappings.

        The following state transition table shows how `self.nameids` ("ids")
        and `self.nametypes` ("types") change with new input (a call to this
        method), and what actions are performed ("implicit"-type system
        messages are INFO/1, and "explicit"-type system messages are ERROR/3):

        ====  =====  ========  ========  =======  ====  =====  =====
         Old State    Input          Action        New State   Notes
        -----------  --------  -----------------  -----------  -----
        ids   types  new type  sys.msg.  dupname  ids   types
        ====  =====  ========  ========  =======  ====  =====  =====
        -     -      explicit  -         -        new   True
        -     -      implicit  -         -        new   False
        None  False  explicit  -         -        new   True
        old   False  explicit  implicit  old      new   True
        None  True   explicit  explicit  new      None  True
        old   True   explicit  explicit  new,old  None  True   [#]_
        None  False  implicit  implicit  new      None  False
        old   False  implicit  implicit  new,old  None  False
        None  True   implicit  implicit  new      None  True
        old   True   implicit  implicit  new      old   True
        ====  =====  ========  ========  =======  ====  =====  =====

        .. [#] Do not clear the name-to-id map or invalidate the old target if
           both old and new targets are external and refer to identical URIs.
           The new target is invalidated regardless.
        """
        for name in node['names']:
            if name in self.nameids:
                self.set_duplicate_name_id(node, id, name, msgnode, explicit)
            else:
                self.nameids[name] = id
                self.nametypes[name] = explicit

    def set_duplicate_name_id(self, node, id, name, msgnode, explicit):
        old_id = self.nameids[name]
        old_explicit = self.nametypes[name]
        self.nametypes[name] = old_explicit or explicit
        if explicit:
            if old_explicit:
                level = 2
                if old_id is not None:
                    old_node = self.ids[old_id]
                    if 'refuri' in node:
                        refuri = node['refuri']
                        if old_node['names'] and 'refuri' in old_node and (old_node['refuri'] == refuri):
                            level = 1
                    if level > 1:
                        dupname(old_node, name)
                        self.nameids[name] = None
                msg = self.reporter.system_message(level, 'Duplicate explicit target name: "%s".' % name, backrefs=[id], base_node=node)
                if msgnode != None:
                    msgnode += msg
                dupname(node, name)
            else:
                self.nameids[name] = id
                if old_id is not None:
                    old_node = self.ids[old_id]
                    dupname(old_node, name)
        else:
            if old_id is not None and (not old_explicit):
                self.nameids[name] = None
                old_node = self.ids[old_id]
                dupname(old_node, name)
            dupname(node, name)
        if not explicit or (not old_explicit and old_id is not None):
            msg = self.reporter.info('Duplicate implicit target name: "%s".' % name, backrefs=[id], base_node=node)
            if msgnode != None:
                msgnode += msg

    def has_name(self, name):
        return name in self.nameids

    def note_implicit_target(self, target, msgnode=None):
        id = self.set_id(target, msgnode)
        self.set_name_id_map(target, id, msgnode, explicit=None)

    def note_explicit_target(self, target, msgnode=None):
        id = self.set_id(target, msgnode)
        self.set_name_id_map(target, id, msgnode, explicit=True)

    def note_refname(self, node):
        self.refnames.setdefault(node['refname'], []).append(node)

    def note_refid(self, node):
        self.refids.setdefault(node['refid'], []).append(node)

    def note_indirect_target(self, target):
        self.indirect_targets.append(target)
        if target['names']:
            self.note_refname(target)

    def note_anonymous_target(self, target):
        self.set_id(target)

    def note_autofootnote(self, footnote):
        self.set_id(footnote)
        self.autofootnotes.append(footnote)

    def note_autofootnote_ref(self, ref):
        self.set_id(ref)
        self.autofootnote_refs.append(ref)

    def note_symbol_footnote(self, footnote):
        self.set_id(footnote)
        self.symbol_footnotes.append(footnote)

    def note_symbol_footnote_ref(self, ref):
        self.set_id(ref)
        self.symbol_footnote_refs.append(ref)

    def note_footnote(self, footnote):
        self.set_id(footnote)
        self.footnotes.append(footnote)

    def note_footnote_ref(self, ref):
        self.set_id(ref)
        self.footnote_refs.setdefault(ref['refname'], []).append(ref)
        self.note_refname(ref)

    def note_citation(self, citation):
        self.citations.append(citation)

    def note_citation_ref(self, ref):
        self.set_id(ref)
        self.citation_refs.setdefault(ref['refname'], []).append(ref)
        self.note_refname(ref)

    def note_substitution_def(self, subdef, def_name, msgnode=None):
        name = whitespace_normalize_name(def_name)
        if name in self.substitution_defs:
            msg = self.reporter.error('Duplicate substitution definition name: "%s".' % name, base_node=subdef)
            if msgnode != None:
                msgnode += msg
            oldnode = self.substitution_defs[name]
            dupname(oldnode, name)
        self.substitution_defs[name] = subdef
        self.substitution_names[fully_normalize_name(name)] = name

    def note_substitution_ref(self, subref, refname):
        subref['refname'] = whitespace_normalize_name(refname)

    def note_pending(self, pending, priority=None):
        self.transformer.add_pending(pending, priority)

    def note_parse_message(self, message):
        self.parse_messages.append(message)

    def note_transform_message(self, message):
        self.transform_messages.append(message)

    def note_source(self, source, offset):
        self.current_source = source
        if offset is None:
            self.current_line = offset
        else:
            self.current_line = offset + 1

    def copy(self):
        obj = self.__class__(self.settings, self.reporter, **self.attributes)
        obj.source = self.source
        obj.line = self.line
        return obj

    def get_decoration(self):
        if not self.decoration:
            self.decoration = decoration()
            index = self.first_child_not_matching_class(Titular)
            if index is None:
                self.append(self.decoration)
            else:
                self.insert(index, self.decoration)
        return self.decoration