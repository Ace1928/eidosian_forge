import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class Footnotes(Transform):
    """
    Assign numbers to autonumbered footnotes, and resolve links to footnotes,
    citations, and their references.

    Given the following ``document`` as input::

        <document>
            <paragraph>
                A labeled autonumbered footnote referece:
                <footnote_reference auto="1" id="id1" refname="footnote">
            <paragraph>
                An unlabeled autonumbered footnote referece:
                <footnote_reference auto="1" id="id2">
            <footnote auto="1" id="id3">
                <paragraph>
                    Unlabeled autonumbered footnote.
            <footnote auto="1" id="footnote" name="footnote">
                <paragraph>
                    Labeled autonumbered footnote.

    Auto-numbered footnotes have attribute ``auto="1"`` and no label.
    Auto-numbered footnote_references have no reference text (they're
    empty elements). When resolving the numbering, a ``label`` element
    is added to the beginning of the ``footnote``, and reference text
    to the ``footnote_reference``.

    The transformed result will be::

        <document>
            <paragraph>
                A labeled autonumbered footnote referece:
                <footnote_reference auto="1" id="id1" refid="footnote">
                    2
            <paragraph>
                An unlabeled autonumbered footnote referece:
                <footnote_reference auto="1" id="id2" refid="id3">
                    1
            <footnote auto="1" id="id3" backrefs="id2">
                <label>
                    1
                <paragraph>
                    Unlabeled autonumbered footnote.
            <footnote auto="1" id="footnote" name="footnote" backrefs="id1">
                <label>
                    2
                <paragraph>
                    Labeled autonumbered footnote.

    Note that the footnotes are not in the same order as the references.

    The labels and reference text are added to the auto-numbered ``footnote``
    and ``footnote_reference`` elements.  Footnote elements are backlinked to
    their references via "refids" attributes.  References are assigned "id"
    and "refid" attributes.

    After adding labels and reference text, the "auto" attributes can be
    ignored.
    """
    default_priority = 620
    autofootnote_labels = None
    'Keep track of unlabeled autonumbered footnotes.'
    symbols = ['*', '†', '‡', '§', '¶', '#', '♠', '♥', '♦', '♣']

    def apply(self):
        self.autofootnote_labels = []
        startnum = self.document.autofootnote_start
        self.document.autofootnote_start = self.number_footnotes(startnum)
        self.number_footnote_references(startnum)
        self.symbolize_footnotes()
        self.resolve_footnotes_and_citations()

    def number_footnotes(self, startnum):
        """
        Assign numbers to autonumbered footnotes.

        For labeled autonumbered footnotes, copy the number over to
        corresponding footnote references.
        """
        for footnote in self.document.autofootnotes:
            while True:
                label = str(startnum)
                startnum += 1
                if label not in self.document.nameids:
                    break
            footnote.insert(0, nodes.label('', label))
            for name in footnote['names']:
                for ref in self.document.footnote_refs.get(name, []):
                    ref += nodes.Text(label)
                    ref.delattr('refname')
                    assert len(footnote['ids']) == len(ref['ids']) == 1
                    ref['refid'] = footnote['ids'][0]
                    footnote.add_backref(ref['ids'][0])
                    self.document.note_refid(ref)
                    ref.resolved = 1
            if not footnote['names'] and (not footnote['dupnames']):
                footnote['names'].append(label)
                self.document.note_explicit_target(footnote, footnote)
                self.autofootnote_labels.append(label)
        return startnum

    def number_footnote_references(self, startnum):
        """Assign numbers to autonumbered footnote references."""
        i = 0
        for ref in self.document.autofootnote_refs:
            if ref.resolved or ref.hasattr('refid'):
                continue
            try:
                label = self.autofootnote_labels[i]
            except IndexError:
                msg = self.document.reporter.error('Too many autonumbered footnote references: only %s corresponding footnotes available.' % len(self.autofootnote_labels), base_node=ref)
                msgid = self.document.set_id(msg)
                for ref in self.document.autofootnote_refs[i:]:
                    if ref.resolved or ref.hasattr('refname'):
                        continue
                    prb = nodes.problematic(ref.rawsource, ref.rawsource, refid=msgid)
                    prbid = self.document.set_id(prb)
                    msg.add_backref(prbid)
                    ref.replace_self(prb)
                break
            ref += nodes.Text(label)
            id = self.document.nameids[label]
            footnote = self.document.ids[id]
            ref['refid'] = id
            self.document.note_refid(ref)
            assert len(ref['ids']) == 1
            footnote.add_backref(ref['ids'][0])
            ref.resolved = 1
            i += 1

    def symbolize_footnotes(self):
        """Add symbols indexes to "[*]"-style footnotes and references."""
        labels = []
        for footnote in self.document.symbol_footnotes:
            reps, index = divmod(self.document.symbol_footnote_start, len(self.symbols))
            labeltext = self.symbols[index] * (reps + 1)
            labels.append(labeltext)
            footnote.insert(0, nodes.label('', labeltext))
            self.document.symbol_footnote_start += 1
            self.document.set_id(footnote)
        i = 0
        for ref in self.document.symbol_footnote_refs:
            try:
                ref += nodes.Text(labels[i])
            except IndexError:
                msg = self.document.reporter.error('Too many symbol footnote references: only %s corresponding footnotes available.' % len(labels), base_node=ref)
                msgid = self.document.set_id(msg)
                for ref in self.document.symbol_footnote_refs[i:]:
                    if ref.resolved or ref.hasattr('refid'):
                        continue
                    prb = nodes.problematic(ref.rawsource, ref.rawsource, refid=msgid)
                    prbid = self.document.set_id(prb)
                    msg.add_backref(prbid)
                    ref.replace_self(prb)
                break
            footnote = self.document.symbol_footnotes[i]
            assert len(footnote['ids']) == 1
            ref['refid'] = footnote['ids'][0]
            self.document.note_refid(ref)
            footnote.add_backref(ref['ids'][0])
            i += 1

    def resolve_footnotes_and_citations(self):
        """
        Link manually-labeled footnotes and citations to/from their
        references.
        """
        for footnote in self.document.footnotes:
            for label in footnote['names']:
                if label in self.document.footnote_refs:
                    reflist = self.document.footnote_refs[label]
                    self.resolve_references(footnote, reflist)
        for citation in self.document.citations:
            for label in citation['names']:
                if label in self.document.citation_refs:
                    reflist = self.document.citation_refs[label]
                    self.resolve_references(citation, reflist)

    def resolve_references(self, note, reflist):
        assert len(note['ids']) == 1
        id = note['ids'][0]
        for ref in reflist:
            if ref.resolved:
                continue
            ref.delattr('refname')
            ref['refid'] = id
            assert len(ref['ids']) == 1
            note.add_backref(ref['ids'][0])
            ref.resolved = 1
        note.resolved = 1