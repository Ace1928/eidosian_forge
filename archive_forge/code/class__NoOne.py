from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
class _NoOne(object):

    def id(self):
        return 'noone'

    def domain(self):
        return ''