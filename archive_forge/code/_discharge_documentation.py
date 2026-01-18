import macaroonbakery.bakery as bakery
import macaroonbakery._utils as utils
Handles a discharge request as received by the /discharge
    endpoint.
    @param ctx The context passed to the checker {checkers.AuthContext}
    @param content URL and form parameters {dict}
    @param locator Locator used to add third party caveats returned by
    the checker {macaroonbakery.ThirdPartyLocator}
    @param checker {macaroonbakery.ThirdPartyCaveatChecker} Used to check third
    party caveats.
    @return The discharge macaroon {macaroonbakery.Macaroon}
    