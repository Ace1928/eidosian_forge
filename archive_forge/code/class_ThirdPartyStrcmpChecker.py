from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
class ThirdPartyStrcmpChecker(bakery.ThirdPartyCaveatChecker):

    def __init__(self, str):
        self.str = str

    def check_third_party_caveat(self, ctx, cav_info):
        condition = cav_info.condition
        if isinstance(cav_info.condition, bytes):
            condition = cav_info.condition.decode('utf-8')
        if condition != self.str:
            raise bakery.ThirdPartyCaveatCheckFailed("{} doesn't match {}".format(repr(condition), repr(self.str)))
        return []