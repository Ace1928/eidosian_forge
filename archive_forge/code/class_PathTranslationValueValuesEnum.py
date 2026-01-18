from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PathTranslationValueValuesEnum(_messages.Enum):
    """PathTranslationValueValuesEnum enum type.

    Values:
      PATH_TRANSLATION_UNSPECIFIED: <no description>
      CONSTANT_ADDRESS: Use the backend address as-is, with no modification to
        the path. If the URL pattern contains variables, the variable names
        and values will be appended to the query string. If a query string
        parameter and a URL pattern variable have the same name, this may
        result in duplicate keys in the query string. # Examples Given the
        following operation config: Method path: /api/company/{cid}/user/{uid}
        Backend address: https://example.cloudfunctions.net/getUser Requests
        to the following request paths will call the backend at the translated
        path: Request path: /api/company/widgetworks/user/johndoe Translated:
        https://example.cloudfunctions.net/getUser?cid=widgetworks&uid=johndoe
        Request path: /api/company/widgetworks/user/johndoe?timezone=EST
        Translated: https://example.cloudfunctions.net/getUser?timezone=EST&ci
        d=widgetworks&uid=johndoe
      APPEND_PATH_TO_ADDRESS: The request path will be appended to the backend
        address. # Examples Given the following operation config: Method path:
        /api/company/{cid}/user/{uid} Backend address:
        https://example.appspot.com Requests to the following request paths
        will call the backend at the translated path: Request path:
        /api/company/widgetworks/user/johndoe Translated:
        https://example.appspot.com/api/company/widgetworks/user/johndoe
        Request path: /api/company/widgetworks/user/johndoe?timezone=EST
        Translated: https://example.appspot.com/api/company/widgetworks/user/j
        ohndoe?timezone=EST
    """
    PATH_TRANSLATION_UNSPECIFIED = 0
    CONSTANT_ADDRESS = 1
    APPEND_PATH_TO_ADDRESS = 2