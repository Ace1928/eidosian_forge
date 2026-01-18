from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
class Visited:
    in_f = False
    in_allow = False
    in_get_acl = False