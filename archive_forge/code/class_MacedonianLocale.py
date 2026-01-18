import sys
from math import trunc
from typing import (
class MacedonianLocale(SlavicBaseLocale):
    names = ['mk', 'mk-mk']
    past = 'пред {0}'
    future = 'за {0}'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'сега', 'second': 'една секунда', 'seconds': {'singular': '{0} секунда', 'dual': '{0} секунди', 'plural': '{0} секунди'}, 'minute': 'една минута', 'minutes': {'singular': '{0} минута', 'dual': '{0} минути', 'plural': '{0} минути'}, 'hour': 'еден саат', 'hours': {'singular': '{0} саат', 'dual': '{0} саати', 'plural': '{0} саати'}, 'day': 'еден ден', 'days': {'singular': '{0} ден', 'dual': '{0} дена', 'plural': '{0} дена'}, 'week': 'една недела', 'weeks': {'singular': '{0} недела', 'dual': '{0} недели', 'plural': '{0} недели'}, 'month': 'еден месец', 'months': {'singular': '{0} месец', 'dual': '{0} месеци', 'plural': '{0} месеци'}, 'year': 'една година', 'years': {'singular': '{0} година', 'dual': '{0} години', 'plural': '{0} години'}}
    meridians = {'am': 'дп', 'pm': 'пп', 'AM': 'претпладне', 'PM': 'попладне'}
    month_names = ['', 'Јануари', 'Февруари', 'Март', 'Април', 'Мај', 'Јуни', 'Јули', 'Август', 'Септември', 'Октомври', 'Ноември', 'Декември']
    month_abbreviations = ['', 'Јан', 'Фев', 'Мар', 'Апр', 'Мај', 'Јун', 'Јул', 'Авг', 'Септ', 'Окт', 'Ноем', 'Декем']
    day_names = ['', 'Понеделник', 'Вторник', 'Среда', 'Четврток', 'Петок', 'Сабота', 'Недела']
    day_abbreviations = ['', 'Пон', 'Вт', 'Сре', 'Чет', 'Пет', 'Саб', 'Нед']