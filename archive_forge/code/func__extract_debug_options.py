import json
import urllib.parse as urllib_parse
def _extract_debug_options(opts, flags=None):
    """Return the debug options encoded in the given value.

    "opts" is a semicolon-separated string of "key=value" pairs.
    "flags" is a list of strings.

    If flags is provided then it is used as a fallback.

    The values come from the launch config:

     {
         type:'python',
         request:'launch'|'attach',
         name:'friendly name for debug config',
         debugOptions:[
             'RedirectOutput', 'Django'
         ],
         options:'REDIRECT_OUTPUT=True;DJANGO_DEBUG=True'
     }

    Further information can be found here:

    https://code.visualstudio.com/docs/editor/debugging#_launchjson-attributes
    """
    if not opts:
        opts = _build_debug_options(flags)
    return _parse_debug_options(opts)