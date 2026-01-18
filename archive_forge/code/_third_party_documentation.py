from collections import namedtuple
import macaroonbakery.checkers as checkers
 ThirdPartyInfo holds information on a given third party
    discharge service.
    @param version The latest bakery protocol version supported
    by the discharger {number}
    @param public_key Public key of the third party {PublicKey}
    