import sys
from math import trunc
from typing import (
class UkrainianLocale(SlavicBaseLocale):
    names = ['ua', 'uk', 'uk-ua']
    past = '{0} тому'
    future = 'за {0}'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'зараз', 'second': 'секунда', 'seconds': '{0} кілька секунд', 'minute': 'хвилину', 'minutes': {'singular': '{0} хвилину', 'dual': '{0} хвилини', 'plural': '{0} хвилин'}, 'hour': 'годину', 'hours': {'singular': '{0} годину', 'dual': '{0} години', 'plural': '{0} годин'}, 'day': 'день', 'days': {'singular': '{0} день', 'dual': '{0} дні', 'plural': '{0} днів'}, 'month': 'місяць', 'months': {'singular': '{0} місяць', 'dual': '{0} місяці', 'plural': '{0} місяців'}, 'year': 'рік', 'years': {'singular': '{0} рік', 'dual': '{0} роки', 'plural': '{0} років'}}
    month_names = ['', 'січня', 'лютого', 'березня', 'квітня', 'травня', 'червня', 'липня', 'серпня', 'вересня', 'жовтня', 'листопада', 'грудня']
    month_abbreviations = ['', 'січ', 'лют', 'бер', 'квіт', 'трав', 'черв', 'лип', 'серп', 'вер', 'жовт', 'лист', 'груд']
    day_names = ['', 'понеділок', 'вівторок', 'середа', 'четвер', 'п’ятниця', 'субота', 'неділя']
    day_abbreviations = ['', 'пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'нд']