import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
class YFuncStats(YStatsIndexable):
    _idx_max = 0
    _sort_type = None
    _sort_order = None
    _SUPPORTED_LOAD_FORMATS = ['YSTAT']
    _SUPPORTED_SAVE_FORMATS = ['YSTAT', 'CALLGRIND', 'PSTAT']

    def __init__(self, files=[]):
        super().__init__()
        self.add(files)
        self._filter_callback = None

    def strip_dirs(self):
        for stat in self:
            stat.strip_dirs()
            stat.children.strip_dirs()
        return self

    def get(self, filter={}, filter_callback=None):
        _yappi._pause()
        self.clear()
        try:
            self._filter_callback = filter_callback
            _yappi.enum_func_stats(self._enumerator, filter)
            self._filter_callback = None
            for stat in self:
                _childs = YChildFuncStats()
                for child_tpl in stat.children:
                    rstat = self[child_tpl[0]]
                    if rstat is None:
                        continue
                    tavg = rstat.ttot / rstat.ncall
                    cfstat = YChildFuncStat(child_tpl + (tavg, rstat.builtin, rstat.full_name, rstat.module, rstat.lineno, rstat.name))
                    _childs.append(cfstat)
                stat.children = _childs
            result = super().get()
        finally:
            _yappi._resume()
        return result

    def _enumerator(self, stat_entry):
        global _fn_descriptor_dict
        fname, fmodule, flineno, fncall, fnactualcall, fbuiltin, fttot, ftsub, findex, fchildren, fctxid, fctxname, ftag, ffn_descriptor = stat_entry
        ffull_name = _func_fullname(bool(fbuiltin), fmodule, flineno, fname)
        ftavg = fttot / fncall
        fstat = YFuncStat(stat_entry + (ftavg, ffull_name))
        _fn_descriptor_dict[ffull_name] = ffn_descriptor
        if os.path.basename(fstat.module) == 'yappi.py' or fstat.module == '_yappi':
            return
        fstat.builtin = bool(fstat.builtin)
        if self._filter_callback:
            if not self._filter_callback(fstat):
                return
        self.append(fstat)
        if self._idx_max < fstat.index:
            self._idx_max = fstat.index

    def _add_from_YSTAT(self, file):
        try:
            saved_stats, saved_clock_type = pickle.load(file)
        except:
            raise YappiError(f'Unable to load the saved profile information from {file.name}.')
        if not self.empty():
            if self._clock_type != saved_clock_type and self._clock_type is not None:
                raise YappiError('Clock type mismatch between current and saved profiler sessions.[%s,%s]' % (self._clock_type, saved_clock_type))
        self._clock_type = saved_clock_type
        for saved_stat in saved_stats:
            if saved_stat not in self:
                self._idx_max += 1
                saved_stat.index = self._idx_max
                self.append(saved_stat)
        for saved_stat in saved_stats:
            for saved_child_stat in saved_stat.children:
                saved_child_stat.index = self[saved_child_stat.full_name].index
        for saved_stat in saved_stats:
            saved_stat_in_curr = self[saved_stat.full_name]
            saved_stat_in_curr += saved_stat

    def _save_as_YSTAT(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self, self._clock_type), f, YPICKLE_PROTOCOL)

    def _save_as_PSTAT(self, path):
        """
        Save the profiling information as PSTAT.
        """
        _stats = convert2pstats(self)
        _stats.dump_stats(path)

    def _save_as_CALLGRIND(self, path):
        """
        Writes all the function stats in a callgrind-style format to the given
        file. (stdout by default)
        """
        header = 'version: 1\ncreator: %s\npid: %d\ncmd:  %s\npart: 1\n\nevents: Ticks' % ('yappi', os.getpid(), ' '.join(sys.argv))
        lines = [header]
        file_ids = ['']
        func_ids = ['']
        func_idx_list = []
        for func_stat in self:
            file_ids += ['fl=(%d) %s' % (func_stat.index, func_stat.module)]
            func_ids += ['fn=(%d) %s %s:%s' % (func_stat.index, func_stat.name, func_stat.module, func_stat.lineno)]
            func_idx_list.append(func_stat.index)
            for child in func_stat.children:
                if child.index in func_idx_list:
                    continue
                file_ids += ['fl=(%d) %s' % (child.index, child.module)]
                func_ids += ['fn=(%d) %s %s:%s' % (child.index, child.name, child.module, child.lineno)]
                func_idx_list.append(child.index)
        lines += file_ids + func_ids
        for func_stat in self:
            func_stats = ['', 'fl=(%d)' % func_stat.index, 'fn=(%d)' % func_stat.index]
            func_stats += [f'{func_stat.lineno} {int(func_stat.tsub * 1000000.0)}']
            for child in func_stat.children:
                func_stats += ['cfl=(%d)' % child.index, 'cfn=(%d)' % child.index, 'calls=%d 0' % child.ncall, '0 %d' % int(child.ttot * 1000000.0)]
            lines += func_stats
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def add(self, files, type='ystat'):
        type = type.upper()
        if type not in self._SUPPORTED_LOAD_FORMATS:
            raise NotImplementedError('Loading from (%s) format is not possible currently.')
        if isinstance(files, str):
            files = [files]
        for fd in files:
            with open(fd, 'rb') as f:
                add_func = getattr(self, f'_add_from_{type}')
                add_func(file=f)
        return self.sort(DEFAULT_SORT_TYPE, DEFAULT_SORT_ORDER)

    def save(self, path, type='ystat'):
        type = type.upper()
        if type not in self._SUPPORTED_SAVE_FORMATS:
            raise NotImplementedError(f'Saving in "{type}" format is not possible currently.')
        save_func = getattr(self, f'_save_as_{type}')
        save_func(path=path)

    def print_all(self, out=sys.stdout, columns={0: ('name', 36), 1: ('ncall', 5), 2: ('tsub', 8), 3: ('ttot', 8), 4: ('tavg', 8)}):
        """
        Prints all of the function profiler results to a given file. (stdout by default)
        """
        if self.empty():
            return
        for _, col in columns.items():
            _validate_columns(col[0], COLUMNS_FUNCSTATS)
        out.write(LINESEP)
        out.write(f'Clock type: {self._clock_type.upper()}')
        out.write(LINESEP)
        out.write(f'Ordered by: {self._sort_type}, {self._sort_order}')
        out.write(LINESEP)
        out.write(LINESEP)
        self._print_header(out, columns)
        for stat in self:
            stat._print(out, columns)

    def sort(self, sort_type, sort_order='desc'):
        sort_type = _validate_sorttype(sort_type, SORT_TYPES_FUNCSTATS)
        sort_order = _validate_sortorder(sort_order)
        self._sort_type = sort_type
        self._sort_order = sort_order
        return super().sort(SORT_TYPES_FUNCSTATS[sort_type], SORT_ORDERS[sort_order])

    def debug_print(self):
        if self.empty():
            return
        console = sys.stdout
        CHILD_STATS_LEFT_MARGIN = 5
        for stat in self:
            console.write('index: %d' % stat.index)
            console.write(LINESEP)
            console.write(f'full_name: {stat.full_name}')
            console.write(LINESEP)
            console.write('ncall: %d/%d' % (stat.ncall, stat.nactualcall))
            console.write(LINESEP)
            console.write(f'ttot: {_fft(stat.ttot)}')
            console.write(LINESEP)
            console.write(f'tsub: {_fft(stat.tsub)}')
            console.write(LINESEP)
            console.write('children: ')
            console.write(LINESEP)
            for child_stat in stat.children:
                console.write(LINESEP)
                console.write(' ' * CHILD_STATS_LEFT_MARGIN)
                console.write('index: %d' % child_stat.index)
                console.write(LINESEP)
                console.write(' ' * CHILD_STATS_LEFT_MARGIN)
                console.write(f'child_full_name: {child_stat.full_name}')
                console.write(LINESEP)
                console.write(' ' * CHILD_STATS_LEFT_MARGIN)
                console.write('ncall: %d/%d' % (child_stat.ncall, child_stat.nactualcall))
                console.write(LINESEP)
                console.write(' ' * CHILD_STATS_LEFT_MARGIN)
                console.write(f'ttot: {_fft(child_stat.ttot)}')
                console.write(LINESEP)
                console.write(' ' * CHILD_STATS_LEFT_MARGIN)
                console.write(f'tsub: {_fft(child_stat.tsub)}')
                console.write(LINESEP)
            console.write(LINESEP)