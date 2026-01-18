import sys
from math import trunc
from typing import (
class SlovenianLocale(Locale):
    names = ['sl', 'sl-si']
    past = 'pred {0}'
    future = 'čez {0}'
    and_word = 'in'
    timeframes = {'now': 'zdaj', 'second': 'sekundo', 'seconds': '{0} sekund', 'minute': 'minuta', 'minutes': '{0} minutami', 'hour': 'uro', 'hours': '{0} ur', 'day': 'dan', 'days': '{0} dni', 'month': 'mesec', 'months': '{0} mesecev', 'year': 'leto', 'years': '{0} let'}
    meridians = {'am': '', 'pm': '', 'AM': '', 'PM': ''}
    month_names = ['', 'Januar', 'Februar', 'Marec', 'April', 'Maj', 'Junij', 'Julij', 'Avgust', 'September', 'Oktober', 'November', 'December']
    month_abbreviations = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun', 'Jul', 'Avg', 'Sep', 'Okt', 'Nov', 'Dec']
    day_names = ['', 'Ponedeljek', 'Torek', 'Sreda', 'Četrtek', 'Petek', 'Sobota', 'Nedelja']
    day_abbreviations = ['', 'Pon', 'Tor', 'Sre', 'Čet', 'Pet', 'Sob', 'Ned']