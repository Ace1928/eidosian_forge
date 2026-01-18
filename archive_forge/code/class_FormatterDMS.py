import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
class FormatterDMS:
    deg_mark = '^{\\circ}'
    min_mark = '^{\\prime}'
    sec_mark = '^{\\prime\\prime}'
    fmt_d = '$%d' + deg_mark + '$'
    fmt_ds = '$%d.%s' + deg_mark + '$'
    fmt_d_m = '$%s%d' + deg_mark + '\\,%02d' + min_mark + '$'
    fmt_d_ms = '$%s%d' + deg_mark + '\\,%02d.%s' + min_mark + '$'
    fmt_d_m_partial = '$%s%d' + deg_mark + '\\,%02d' + min_mark + '\\,'
    fmt_s_partial = '%02d' + sec_mark + '$'
    fmt_ss_partial = '%02d.%s' + sec_mark + '$'

    def _get_number_fraction(self, factor):
        number_fraction = None
        for threshold in [1, 60, 3600]:
            if factor <= threshold:
                break
            d = factor // threshold
            int_log_d = int(np.floor(np.log10(d)))
            if 10 ** int_log_d == d and d != 1:
                number_fraction = int_log_d
                factor = factor // 10 ** int_log_d
                return (factor, number_fraction)
        return (factor, number_fraction)

    def __call__(self, direction, factor, values):
        if len(values) == 0:
            return []
        ss = np.sign(values)
        signs = ['-' if v < 0 else '' for v in values]
        factor, number_fraction = self._get_number_fraction(factor)
        values = np.abs(values)
        if number_fraction is not None:
            values, frac_part = divmod(values, 10 ** number_fraction)
            frac_fmt = '%%0%dd' % (number_fraction,)
            frac_str = [frac_fmt % (f1,) for f1 in frac_part]
        if factor == 1:
            if number_fraction is None:
                return [self.fmt_d % (s * int(v),) for s, v in zip(ss, values)]
            else:
                return [self.fmt_ds % (s * int(v), f1) for s, v, f1 in zip(ss, values, frac_str)]
        elif factor == 60:
            deg_part, min_part = divmod(values, 60)
            if number_fraction is None:
                return [self.fmt_d_m % (s1, d1, m1) for s1, d1, m1 in zip(signs, deg_part, min_part)]
            else:
                return [self.fmt_d_ms % (s, d1, m1, f1) for s, d1, m1, f1 in zip(signs, deg_part, min_part, frac_str)]
        elif factor == 3600:
            if ss[-1] == -1:
                inverse_order = True
                values = values[::-1]
                signs = signs[::-1]
            else:
                inverse_order = False
            l_hm_old = ''
            r = []
            deg_part, min_part_ = divmod(values, 3600)
            min_part, sec_part = divmod(min_part_, 60)
            if number_fraction is None:
                sec_str = [self.fmt_s_partial % (s1,) for s1 in sec_part]
            else:
                sec_str = [self.fmt_ss_partial % (s1, f1) for s1, f1 in zip(sec_part, frac_str)]
            for s, d1, m1, s1 in zip(signs, deg_part, min_part, sec_str):
                l_hm = self.fmt_d_m_partial % (s, d1, m1)
                if l_hm != l_hm_old:
                    l_hm_old = l_hm
                    l = l_hm + s1
                else:
                    l = '$' + s + s1
                r.append(l)
            if inverse_order:
                return r[::-1]
            else:
                return r
        else:
            return ['$%s^{\\circ}$' % v for v in ss * values]