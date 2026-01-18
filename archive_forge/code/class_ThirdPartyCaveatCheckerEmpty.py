from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
class ThirdPartyCaveatCheckerEmpty(bakery.ThirdPartyCaveatChecker):

    def check_third_party_caveat(self, ctx, cav_info):
        return []