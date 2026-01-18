from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2314
from pyasn1_modules import rfc2459
from pyasn1_modules import rfc2511
class PKIBody(univ.Choice):
    """
    PKIBody ::= CHOICE {       -- message-specific body elements
         ir       [0]  CertReqMessages,        --Initialization Request
         ip       [1]  CertRepMessage,         --Initialization Response
         cr       [2]  CertReqMessages,        --Certification Request
         cp       [3]  CertRepMessage,         --Certification Response
         p10cr    [4]  CertificationRequest,   --imported from [PKCS10]
         popdecc  [5]  POPODecKeyChallContent, --pop Challenge
         popdecr  [6]  POPODecKeyRespContent,  --pop Response
         kur      [7]  CertReqMessages,        --Key Update Request
         kup      [8]  CertRepMessage,         --Key Update Response
         krr      [9]  CertReqMessages,        --Key Recovery Request
         krp      [10] KeyRecRepContent,       --Key Recovery Response
         rr       [11] RevReqContent,          --Revocation Request
         rp       [12] RevRepContent,          --Revocation Response
         ccr      [13] CertReqMessages,        --Cross-Cert. Request
         ccp      [14] CertRepMessage,         --Cross-Cert. Response
         ckuann   [15] CAKeyUpdAnnContent,     --CA Key Update Ann.
         cann     [16] CertAnnContent,         --Certificate Ann.
         rann     [17] RevAnnContent,          --Revocation Ann.
         crlann   [18] CRLAnnContent,          --CRL Announcement
         pkiconf  [19] PKIConfirmContent,      --Confirmation
         nested   [20] NestedMessageContent,   --Nested Message
         genm     [21] GenMsgContent,          --General Message
         genp     [22] GenRepContent,          --General Response
         error    [23] ErrorMsgContent,        --Error Message
         certConf [24] CertConfirmContent,     --Certificate confirm
         pollReq  [25] PollReqContent,         --Polling request
         pollRep  [26] PollRepContent          --Polling response

    """
    componentType = namedtype.NamedTypes(namedtype.NamedType('ir', rfc2511.CertReqMessages().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.NamedType('ip', CertRepMessage().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))), namedtype.NamedType('cr', rfc2511.CertReqMessages().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2))), namedtype.NamedType('cp', CertRepMessage().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3))), namedtype.NamedType('p10cr', rfc2314.CertificationRequest().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4))), namedtype.NamedType('popdecc', POPODecKeyChallContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 5))), namedtype.NamedType('popdecr', POPODecKeyRespContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 6))), namedtype.NamedType('kur', rfc2511.CertReqMessages().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 7))), namedtype.NamedType('kup', CertRepMessage().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 8))), namedtype.NamedType('krr', rfc2511.CertReqMessages().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 9))), namedtype.NamedType('krp', KeyRecRepContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 10))), namedtype.NamedType('rr', RevReqContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 11))), namedtype.NamedType('rp', RevRepContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 12))), namedtype.NamedType('ccr', rfc2511.CertReqMessages().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 13))), namedtype.NamedType('ccp', CertRepMessage().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 14))), namedtype.NamedType('ckuann', CAKeyUpdAnnContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 15))), namedtype.NamedType('cann', CertAnnContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 16))), namedtype.NamedType('rann', RevAnnContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 17))), namedtype.NamedType('crlann', CRLAnnContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 18))), namedtype.NamedType('pkiconf', PKIConfirmContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 19))), namedtype.NamedType('nested', nestedMessageContent), namedtype.NamedType('genm', GenMsgContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 21))), namedtype.NamedType('gen', GenRepContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 22))), namedtype.NamedType('error', ErrorMsgContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 23))), namedtype.NamedType('certConf', CertConfirmContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 24))), namedtype.NamedType('pollReq', PollReqContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 25))), namedtype.NamedType('pollRep', PollRepContent().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 26))))