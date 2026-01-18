import sys
from math import trunc
from typing import (
class BelarusianLocale(SlavicBaseLocale):
    names = ['be', 'be-by']
    past = '{0} таму'
    future = 'праз {0}'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'зараз', 'second': 'секунду', 'seconds': '{0} некалькі секунд', 'minute': 'хвіліну', 'minutes': {'singular': '{0} хвіліну', 'dual': '{0} хвіліны', 'plural': '{0} хвілін'}, 'hour': 'гадзіну', 'hours': {'singular': '{0} гадзіну', 'dual': '{0} гадзіны', 'plural': '{0} гадзін'}, 'day': 'дзень', 'days': {'singular': '{0} дзень', 'dual': '{0} дні', 'plural': '{0} дзён'}, 'month': 'месяц', 'months': {'singular': '{0} месяц', 'dual': '{0} месяцы', 'plural': '{0} месяцаў'}, 'year': 'год', 'years': {'singular': '{0} год', 'dual': '{0} гады', 'plural': '{0} гадоў'}}
    month_names = ['', 'студзеня', 'лютага', 'сакавіка', 'красавіка', 'траўня', 'чэрвеня', 'ліпеня', 'жніўня', 'верасня', 'кастрычніка', 'лістапада', 'снежня']
    month_abbreviations = ['', 'студ', 'лют', 'сак', 'крас', 'трав', 'чэрв', 'ліп', 'жнів', 'вер', 'каст', 'ліст', 'снеж']
    day_names = ['', 'панядзелак', 'аўторак', 'серада', 'чацвер', 'пятніца', 'субота', 'нядзеля']
    day_abbreviations = ['', 'пн', 'ат', 'ср', 'чц', 'пт', 'сб', 'нд']