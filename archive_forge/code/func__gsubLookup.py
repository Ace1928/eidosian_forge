import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _gsubLookup(self, lookup, prefix, suffix, ignore, chain, fealookup):
    statements = fealookup.statements
    sub = lookup.sub
    for key, val in sub.mapping.items():
        if not key or not val:
            path, line, column = sub.location
            log.warning(f'{path}:{line}:{column}: Ignoring empty substitution')
            continue
        statement = None
        glyphs = self._coverage(key)
        replacements = self._coverage(val)
        if ignore:
            chain_context = (prefix, glyphs, suffix)
            statement = ast.IgnoreSubstStatement([chain_context])
        elif isinstance(sub, VAst.SubstitutionSingleDefinition):
            assert len(glyphs) == 1
            assert len(replacements) == 1
            statement = ast.SingleSubstStatement(glyphs, replacements, prefix, suffix, chain)
        elif isinstance(sub, VAst.SubstitutionReverseChainingSingleDefinition):
            assert len(glyphs) == 1
            assert len(replacements) == 1
            statement = ast.ReverseChainSingleSubstStatement(prefix, suffix, glyphs, replacements)
        elif isinstance(sub, VAst.SubstitutionMultipleDefinition):
            assert len(glyphs) == 1
            statement = ast.MultipleSubstStatement(prefix, glyphs[0], suffix, replacements, chain)
        elif isinstance(sub, VAst.SubstitutionLigatureDefinition):
            assert len(replacements) == 1
            statement = ast.LigatureSubstStatement(prefix, glyphs, suffix, replacements[0], chain)
        else:
            raise NotImplementedError(sub)
        statements.append(statement)