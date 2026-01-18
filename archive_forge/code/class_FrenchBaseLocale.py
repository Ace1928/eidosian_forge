import sys
from math import trunc
from typing import (
class FrenchBaseLocale(Locale):
    past = 'il y a {0}'
    future = 'dans {0}'
    and_word = 'et'
    timeframes = {'now': 'maintenant', 'second': 'une seconde', 'seconds': '{0} secondes', 'minute': 'une minute', 'minutes': '{0} minutes', 'hour': 'une heure', 'hours': '{0} heures', 'day': 'un jour', 'days': '{0} jours', 'week': 'une semaine', 'weeks': '{0} semaines', 'month': 'un mois', 'months': '{0} mois', 'year': 'un an', 'years': '{0} ans'}
    month_names = ['', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
    day_names = ['', 'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
    day_abbreviations = ['', 'lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim']
    ordinal_day_re = '((?P<value>\\b1(?=er\\b)|[1-3]?[02-9](?=e\\b)|[1-3]1(?=e\\b))(er|e)\\b)'

    def _ordinal_number(self, n: int) -> str:
        if abs(n) == 1:
            return f'{n}er'
        return f'{n}e'