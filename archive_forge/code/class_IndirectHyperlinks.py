import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class IndirectHyperlinks(Transform):
    """
    a) Indirect external references::

           <paragraph>
               <reference refname="indirect external">
                   indirect external
           <target id="id1" name="direct external"
               refuri="http://indirect">
           <target id="id2" name="indirect external"
               refname="direct external">

       The "refuri" attribute is migrated back to all indirect targets
       from the final direct target (i.e. a target not referring to
       another indirect target)::

           <paragraph>
               <reference refname="indirect external">
                   indirect external
           <target id="id1" name="direct external"
               refuri="http://indirect">
           <target id="id2" name="indirect external"
               refuri="http://indirect">

       Once the attribute is migrated, the preexisting "refname" attribute
       is dropped.

    b) Indirect internal references::

           <target id="id1" name="final target">
           <paragraph>
               <reference refname="indirect internal">
                   indirect internal
           <target id="id2" name="indirect internal 2"
               refname="final target">
           <target id="id3" name="indirect internal"
               refname="indirect internal 2">

       Targets which indirectly refer to an internal target become one-hop
       indirect (their "refid" attributes are directly set to the internal
       target's "id"). References which indirectly refer to an internal
       target become direct internal references::

           <target id="id1" name="final target">
           <paragraph>
               <reference refid="id1">
                   indirect internal
           <target id="id2" name="indirect internal 2" refid="id1">
           <target id="id3" name="indirect internal" refid="id1">
    """
    default_priority = 460

    def apply(self):
        for target in self.document.indirect_targets:
            if not target.resolved:
                self.resolve_indirect_target(target)
            self.resolve_indirect_references(target)

    def resolve_indirect_target(self, target):
        refname = target.get('refname')
        if refname is None:
            reftarget_id = target['refid']
        else:
            reftarget_id = self.document.nameids.get(refname)
            if not reftarget_id:
                for resolver_function in self.document.transformer.unknown_reference_resolvers:
                    if resolver_function(target):
                        break
                else:
                    self.nonexistent_indirect_target(target)
                return
        reftarget = self.document.ids[reftarget_id]
        reftarget.note_referenced_by(id=reftarget_id)
        if isinstance(reftarget, nodes.target) and (not reftarget.resolved) and reftarget.hasattr('refname'):
            if hasattr(target, 'multiply_indirect'):
                self.circular_indirect_reference(target)
                return
            target.multiply_indirect = 1
            self.resolve_indirect_target(reftarget)
            del target.multiply_indirect
        if reftarget.hasattr('refuri'):
            target['refuri'] = reftarget['refuri']
            if 'refid' in target:
                del target['refid']
        elif reftarget.hasattr('refid'):
            target['refid'] = reftarget['refid']
            self.document.note_refid(target)
        elif reftarget['ids']:
            target['refid'] = reftarget_id
            self.document.note_refid(target)
        else:
            self.nonexistent_indirect_target(target)
            return
        if refname is not None:
            del target['refname']
        target.resolved = 1

    def nonexistent_indirect_target(self, target):
        if target['refname'] in self.document.nameids:
            self.indirect_target_error(target, 'which is a duplicate, and cannot be used as a unique reference')
        else:
            self.indirect_target_error(target, 'which does not exist')

    def circular_indirect_reference(self, target):
        self.indirect_target_error(target, 'forming a circular reference')

    def indirect_target_error(self, target, explanation):
        naming = ''
        reflist = []
        if target['names']:
            naming = '"%s" ' % target['names'][0]
        for name in target['names']:
            reflist.extend(self.document.refnames.get(name, []))
        for id in target['ids']:
            reflist.extend(self.document.refids.get(id, []))
        if target['ids']:
            naming += '(id="%s")' % target['ids'][0]
        msg = self.document.reporter.error('Indirect hyperlink target %s refers to target "%s", %s.' % (naming, target['refname'], explanation), base_node=target)
        msgid = self.document.set_id(msg)
        for ref in utils.uniq(reflist):
            prb = nodes.problematic(ref.rawsource, ref.rawsource, refid=msgid)
            prbid = self.document.set_id(prb)
            msg.add_backref(prbid)
            ref.replace_self(prb)
        target.resolved = 1

    def resolve_indirect_references(self, target):
        if target.hasattr('refid'):
            attname = 'refid'
            call_method = self.document.note_refid
        elif target.hasattr('refuri'):
            attname = 'refuri'
            call_method = None
        else:
            return
        attval = target[attname]
        for name in target['names']:
            reflist = self.document.refnames.get(name, [])
            if reflist:
                target.note_referenced_by(name=name)
            for ref in reflist:
                if ref.resolved:
                    continue
                del ref['refname']
                ref[attname] = attval
                if call_method:
                    call_method(ref)
                ref.resolved = 1
                if isinstance(ref, nodes.target):
                    self.resolve_indirect_references(ref)
        for id in target['ids']:
            reflist = self.document.refids.get(id, [])
            if reflist:
                target.note_referenced_by(id=id)
            for ref in reflist:
                if ref.resolved:
                    continue
                del ref['refid']
                ref[attname] = attval
                if call_method:
                    call_method(ref)
                ref.resolved = 1
                if isinstance(ref, nodes.target):
                    self.resolve_indirect_references(ref)