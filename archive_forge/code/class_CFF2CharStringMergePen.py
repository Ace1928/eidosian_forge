from collections import namedtuple
from fontTools.cffLib import (
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial
from .errors import (
class CFF2CharStringMergePen(T2CharStringPen):
    """Pen to merge Type 2 CharStrings."""

    def __init__(self, default_commands, glyphName, num_masters, master_idx, roundTolerance=0.01):
        super().__init__(width=None, glyphSet=None, CFF2=True, roundTolerance=roundTolerance)
        self.pt_index = 0
        self._commands = default_commands
        self.m_index = master_idx
        self.num_masters = num_masters
        self.prev_move_idx = 0
        self.seen_moveto = False
        self.glyphName = glyphName
        self.round = roundFunc(roundTolerance, round=round)

    def add_point(self, point_type, pt_coords):
        if self.m_index == 0:
            self._commands.append([point_type, [pt_coords]])
        else:
            cmd = self._commands[self.pt_index]
            if cmd[0] != point_type:
                raise VarLibCFFPointTypeMergeError(point_type, self.pt_index, len(cmd[1]), cmd[0], self.glyphName)
            cmd[1].append(pt_coords)
        self.pt_index += 1

    def add_hint(self, hint_type, args):
        if self.m_index == 0:
            self._commands.append([hint_type, [args]])
        else:
            cmd = self._commands[self.pt_index]
            if cmd[0] != hint_type:
                raise VarLibCFFHintTypeMergeError(hint_type, self.pt_index, len(cmd[1]), cmd[0], self.glyphName)
            cmd[1].append(args)
        self.pt_index += 1

    def add_hintmask(self, hint_type, abs_args):
        if self.m_index == 0:
            self._commands.append([hint_type, []])
            self._commands.append(['', [abs_args]])
        else:
            cmd = self._commands[self.pt_index]
            if cmd[0] != hint_type:
                raise VarLibCFFHintTypeMergeError(hint_type, self.pt_index, len(cmd[1]), cmd[0], self.glyphName)
            self.pt_index += 1
            cmd = self._commands[self.pt_index]
            cmd[1].append(abs_args)
        self.pt_index += 1

    def _moveTo(self, pt):
        if not self.seen_moveto:
            self.seen_moveto = True
        pt_coords = self._p(pt)
        self.add_point('rmoveto', pt_coords)
        self.prev_move_idx = self.pt_index - 1

    def _lineTo(self, pt):
        pt_coords = self._p(pt)
        self.add_point('rlineto', pt_coords)

    def _curveToOne(self, pt1, pt2, pt3):
        _p = self._p
        pt_coords = _p(pt1) + _p(pt2) + _p(pt3)
        self.add_point('rrcurveto', pt_coords)

    def _closePath(self):
        pass

    def _endPath(self):
        pass

    def restart(self, region_idx):
        self.pt_index = 0
        self.m_index = region_idx
        self._p0 = (0, 0)

    def getCommands(self):
        return self._commands

    def reorder_blend_args(self, commands, get_delta_func):
        """
        We first re-order the master coordinate values.
        For a moveto to lineto, the args are now arranged as::

                [ [master_0 x,y], [master_1 x,y], [master_2 x,y] ]

        We re-arrange this to::

                [	[master_0 x, master_1 x, master_2 x],
                        [master_0 y, master_1 y, master_2 y]
                ]

        If the master values are all the same, we collapse the list to
        as single value instead of a list.

        We then convert this to::

                [ [master_0 x] + [x delta tuple] + [numBlends=1]
                  [master_0 y] + [y delta tuple] + [numBlends=1]
                ]
        """
        for cmd in commands:
            args = cmd[1]
            m_args = zip(*args)
            cmd[1] = list(m_args)
        lastOp = None
        for cmd in commands:
            op = cmd[0]
            if lastOp in ['hintmask', 'cntrmask']:
                coord = list(cmd[1])
                if not allEqual(coord):
                    raise VarLibMergeError('Hintmask values cannot differ between source fonts.')
                cmd[1] = [coord[0][0]]
            else:
                coords = cmd[1]
                new_coords = []
                for coord in coords:
                    if allEqual(coord):
                        new_coords.append(coord[0])
                    else:
                        deltas = get_delta_func(coord)[1:]
                        coord = [coord[0]] + deltas
                        coord.append(1)
                        new_coords.append(coord)
                cmd[1] = new_coords
            lastOp = op
        return commands

    def getCharString(self, private=None, globalSubrs=None, var_model=None, optimize=True):
        commands = self._commands
        commands = self.reorder_blend_args(commands, partial(var_model.getDeltas, round=self.round))
        if optimize:
            commands = specializeCommands(commands, generalizeFirst=False, maxstack=maxStackLimit)
        program = commandsToProgram(commands)
        charString = T2CharString(program=program, private=private, globalSubrs=globalSubrs)
        return charString