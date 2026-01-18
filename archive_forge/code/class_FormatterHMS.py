import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
class FormatterHMS(FormatterDMS):
    deg_mark = '^\\mathrm{h}'
    min_mark = '^\\mathrm{m}'
    sec_mark = '^\\mathrm{s}'
    fmt_d = '$%d' + deg_mark + '$'
    fmt_ds = '$%d.%s' + deg_mark + '$'
    fmt_d_m = '$%s%d' + deg_mark + '\\,%02d' + min_mark + '$'
    fmt_d_ms = '$%s%d' + deg_mark + '\\,%02d.%s' + min_mark + '$'
    fmt_d_m_partial = '$%s%d' + deg_mark + '\\,%02d' + min_mark + '\\,'
    fmt_s_partial = '%02d' + sec_mark + '$'
    fmt_ss_partial = '%02d.%s' + sec_mark + '$'

    def __call__(self, direction, factor, values):
        return super().__call__(direction, factor, np.asarray(values) / 15)