import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _lookupDefinition(self, lookup):
    mark_attachement = None
    mark_filtering = None
    flags = 0
    if lookup.direction == 'RTL':
        flags |= 1
    if not lookup.process_base:
        flags |= 2
    if not lookup.process_marks:
        flags |= 8
    elif isinstance(lookup.process_marks, str):
        mark_attachement = self._groupName(lookup.process_marks)
    elif lookup.mark_glyph_set is not None:
        mark_filtering = self._groupName(lookup.mark_glyph_set)
    lookupflags = None
    if flags or mark_attachement is not None or mark_filtering is not None:
        lookupflags = ast.LookupFlagStatement(flags, mark_attachement, mark_filtering)
    if '\\' in lookup.name:
        name = lookup.name.split('\\')[0]
        if name.lower() not in self._lookups:
            fealookup = ast.LookupBlock(self._lookupName(name))
            if lookupflags is not None:
                fealookup.statements.append(lookupflags)
            fealookup.statements.append(ast.Comment('# ' + lookup.name))
        else:
            fealookup = self._lookups[name.lower()]
            fealookup.statements.append(ast.SubtableStatement())
            fealookup.statements.append(ast.Comment('# ' + lookup.name))
        self._lookups[name.lower()] = fealookup
    else:
        fealookup = ast.LookupBlock(self._lookupName(lookup.name))
        if lookupflags is not None:
            fealookup.statements.append(lookupflags)
        self._lookups[lookup.name.lower()] = fealookup
    if lookup.comments is not None:
        fealookup.statements.append(ast.Comment('# ' + lookup.comments))
    contexts = []
    if lookup.context:
        for context in lookup.context:
            prefix = self._context(context.left)
            suffix = self._context(context.right)
            ignore = context.ex_or_in == 'EXCEPT_CONTEXT'
            contexts.append([prefix, suffix, ignore, False])
            if ignore and len(lookup.context) == 1:
                contexts.append([[], [], False, True])
    else:
        contexts.append([[], [], False, False])
    targetlookup = None
    for prefix, suffix, ignore, chain in contexts:
        if lookup.sub is not None:
            self._gsubLookup(lookup, prefix, suffix, ignore, chain, fealookup)
        if lookup.pos is not None:
            if self._settings.get('COMPILER_USEEXTENSIONLOOKUPS'):
                fealookup.use_extension = True
            if prefix or suffix or chain or ignore:
                if not ignore and targetlookup is None:
                    targetname = self._lookupName(lookup.name + ' target')
                    targetlookup = ast.LookupBlock(targetname)
                    fealookup.targets = getattr(fealookup, 'targets', [])
                    fealookup.targets.append(targetlookup)
                    self._gposLookup(lookup, targetlookup)
                self._gposContextLookup(lookup, prefix, suffix, ignore, fealookup, targetlookup)
            else:
                self._gposLookup(lookup, fealookup)