import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _gposLookup(self, lookup, fealookup):
    statements = fealookup.statements
    pos = lookup.pos
    if isinstance(pos, VAst.PositionAdjustPairDefinition):
        for (idx1, idx2), (pos1, pos2) in pos.adjust_pair.items():
            coverage_1 = pos.coverages_1[idx1 - 1]
            coverage_2 = pos.coverages_2[idx2 - 1]
            enumerated = False
            for item in coverage_1 + coverage_2:
                if not isinstance(item, VAst.GroupName):
                    enumerated = True
            glyphs1 = self._coverage(coverage_1)
            glyphs2 = self._coverage(coverage_2)
            record1 = self._adjustment(pos1)
            record2 = self._adjustment(pos2)
            assert len(glyphs1) == 1
            assert len(glyphs2) == 1
            statements.append(ast.PairPosStatement(glyphs1[0], record1, glyphs2[0], record2, enumerated=enumerated))
    elif isinstance(pos, VAst.PositionAdjustSingleDefinition):
        for a, b in pos.adjust_single:
            glyphs = self._coverage(a)
            record = self._adjustment(b)
            assert len(glyphs) == 1
            statements.append(ast.SinglePosStatement([(glyphs[0], record)], [], [], False))
    elif isinstance(pos, VAst.PositionAttachDefinition):
        anchors = {}
        for marks, classname in pos.coverage_to:
            for mark in marks:
                for name in mark.glyphSet():
                    key = (name, 'MARK_' + classname)
                    self._markclasses[key].used = True
            markclass = ast.MarkClass(self._className(classname))
            for base in pos.coverage:
                for name in base.glyphSet():
                    if name not in anchors:
                        anchors[name] = []
                    if classname not in anchors[name]:
                        anchors[name].append(classname)
        for name in anchors:
            components = 1
            if name in self._ligatures:
                components = self._ligatures[name]
            marks = []
            for mark in anchors[name]:
                markclass = ast.MarkClass(self._className(mark))
                for component in range(1, components + 1):
                    if len(marks) < component:
                        marks.append([])
                    anchor = None
                    if component in self._anchors[name][mark]:
                        anchor = self._anchors[name][mark][component]
                    marks[component - 1].append((anchor, markclass))
            base = self._glyphName(name)
            if name in self._marks:
                mark = ast.MarkMarkPosStatement(base, marks[0])
            elif name in self._ligatures:
                mark = ast.MarkLigPosStatement(base, marks)
            else:
                mark = ast.MarkBasePosStatement(base, marks[0])
            statements.append(mark)
    elif isinstance(pos, VAst.PositionAttachCursiveDefinition):
        enter_coverage = []
        for coverage in pos.coverages_enter:
            for base in coverage:
                for name in base.glyphSet():
                    enter_coverage.append(name)
        exit_coverage = []
        for coverage in pos.coverages_exit:
            for base in coverage:
                for name in base.glyphSet():
                    exit_coverage.append(name)
        for name in enter_coverage:
            glyph = self._glyphName(name)
            entry = self._anchors[name]['entry'][1]
            exit = None
            if name in exit_coverage:
                exit = self._anchors[name]['exit'][1]
                exit_coverage.pop(exit_coverage.index(name))
            statements.append(ast.CursivePosStatement(glyph, entry, exit))
        for name in exit_coverage:
            glyph = self._glyphName(name)
            exit = self._anchors[name]['exit'][1]
            statements.append(ast.CursivePosStatement(glyph, None, exit))
    else:
        raise NotImplementedError(pos)