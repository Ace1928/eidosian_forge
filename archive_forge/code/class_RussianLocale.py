import sys
from math import trunc
from typing import (
class RussianLocale(SlavicBaseLocale):
    names = ['ru', 'ru-ru']
    past = '{0} назад'
    future = 'через {0}'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'сейчас', 'second': 'секунда', 'seconds': {'singular': '{0} секунду', 'dual': '{0} секунды', 'plural': '{0} секунд'}, 'minute': 'минуту', 'minutes': {'singular': '{0} минуту', 'dual': '{0} минуты', 'plural': '{0} минут'}, 'hour': 'час', 'hours': {'singular': '{0} час', 'dual': '{0} часа', 'plural': '{0} часов'}, 'day': 'день', 'days': {'singular': '{0} день', 'dual': '{0} дня', 'plural': '{0} дней'}, 'week': 'неделю', 'weeks': {'singular': '{0} неделю', 'dual': '{0} недели', 'plural': '{0} недель'}, 'month': 'месяц', 'months': {'singular': '{0} месяц', 'dual': '{0} месяца', 'plural': '{0} месяцев'}, 'quarter': 'квартал', 'quarters': {'singular': '{0} квартал', 'dual': '{0} квартала', 'plural': '{0} кварталов'}, 'year': 'год', 'years': {'singular': '{0} год', 'dual': '{0} года', 'plural': '{0} лет'}}
    month_names = ['', 'января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
    month_abbreviations = ['', 'янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
    day_names = ['', 'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
    day_abbreviations = ['', 'пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс']