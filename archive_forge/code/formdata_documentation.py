from .._compat import basestring
from .._compat import urlencode as _urlencode
Handle nested form-data queries and serialize them appropriately.

    There are times when a website expects a nested form data query to be sent
    but, the standard library's urlencode function does not appropriately
    handle the nested structures. In that case, you need this function which
    will flatten the structure first and then properly encode it for you.

    When using this to send data in the body of a request, make sure you
    specify the appropriate Content-Type header for the request.

    .. code-block:: python

        import requests
        from requests_toolbelt.utils import formdata

        query = {
           'my_dict': {
               'foo': 'bar',
               'biz': 'baz",
            },
            'a': 'b',
        }

        resp = requests.get(url, params=formdata.urlencode(query))
        # or
        resp = requests.post(
            url,
            data=formdata.urlencode(query),
            headers={
                'Content-Type': 'application/x-www-form-urlencoded'
            },
        )

    Similarly, you can specify a list of nested tuples, e.g.,

    .. code-block:: python

        import requests
        from requests_toolbelt.utils import formdata

        query = [
            ('my_list', [
                ('foo', 'bar'),
                ('biz', 'baz'),
            ]),
            ('a', 'b'),
        ]

        resp = requests.get(url, params=formdata.urlencode(query))
        # or
        resp = requests.post(
            url,
            data=formdata.urlencode(query),
            headers={
                'Content-Type': 'application/x-www-form-urlencoded'
            },
        )

    For additional parameter and return information, see the official
    `urlencode`_ documentation.

    .. _urlencode:
        https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlencode
    