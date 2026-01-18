import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _buildFeatureFile(self, tables):
    doc = ast.FeatureFile()
    statements = doc.statements
    if self._glyphclasses:
        statements.append(ast.Comment('# Glyph classes'))
        statements.extend(self._glyphclasses.values())
    if self._markclasses:
        statements.append(ast.Comment('\n# Mark classes'))
        statements.extend((c[1] for c in sorted(self._markclasses.items())))
    if self._lookups:
        statements.append(ast.Comment('\n# Lookups'))
        for lookup in self._lookups.values():
            statements.extend(getattr(lookup, 'targets', []))
            statements.append(lookup)
    features = self._features.copy()
    for ftag in features:
        scripts = features[ftag]
        for stag in scripts:
            langs = scripts[stag]
            for ltag in langs:
                langs[ltag] = [l for l in langs[ltag] if l.lower() in self._lookups]
            scripts[stag] = {t: l for t, l in langs.items() if l}
        features[ftag] = {t: s for t, s in scripts.items() if s}
    features = {t: f for t, f in features.items() if f}
    if features:
        statements.append(ast.Comment('# Features'))
        for ftag, scripts in features.items():
            feature = ast.FeatureBlock(ftag)
            stags = sorted(scripts, key=lambda k: 0 if k == 'DFLT' else 1)
            for stag in stags:
                feature.statements.append(ast.ScriptStatement(stag))
                ltags = sorted(scripts[stag], key=lambda k: 0 if k == 'dflt' else 1)
                for ltag in ltags:
                    include_default = True if ltag == 'dflt' else False
                    feature.statements.append(ast.LanguageStatement(ltag, include_default=include_default))
                    for name in scripts[stag][ltag]:
                        lookup = self._lookups[name.lower()]
                        lookupref = ast.LookupReferenceStatement(lookup)
                        feature.statements.append(lookupref)
            statements.append(feature)
    if self._gdef and 'GDEF' in tables:
        classes = []
        for name in ('BASE', 'MARK', 'LIGATURE', 'COMPONENT'):
            if name in self._gdef:
                classname = 'GDEF_' + name.lower()
                glyphclass = ast.GlyphClassDefinition(classname, self._gdef[name])
                statements.append(glyphclass)
                classes.append(ast.GlyphClassName(glyphclass))
            else:
                classes.append(None)
        gdef = ast.TableBlock('GDEF')
        gdef.statements.append(ast.GlyphClassDefStatement(*classes))
        statements.append(gdef)
    return doc