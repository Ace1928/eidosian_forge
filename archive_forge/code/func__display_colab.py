import datetime
import errno
import html
import json
import os
import random
import shlex
import textwrap
import time
from tensorboard import manager
def _display_colab(port, height, display_handle):
    """Display a TensorBoard instance in a Colab output frame.

    The Colab VM is not directly exposed to the network, so the Colab
    runtime provides a service worker tunnel to proxy requests from the
    end user's browser through to servers running on the Colab VM: the
    output frame may issue requests to https://localhost:<port> (HTTPS
    only), which will be forwarded to the specified port on the VM.

    It does not suffice to create an `iframe` and let the service worker
    redirect its traffic (`<iframe src="https://localhost:6006">`),
    because for security reasons service workers cannot intercept iframe
    traffic. Instead, we manually fetch the TensorBoard index page with an
    XHR in the output frame, and inject the raw HTML into `document.body`.

    By default, the TensorBoard web app requests resources against
    relative paths, like `./data/logdir`. Within the output frame, these
    requests must instead hit `https://localhost:<port>/data/logdir`. To
    redirect them, we change the document base URI, which transparently
    affects all requests (XHRs and resources alike).
    """
    import IPython.display
    shell = "\n        (async () => {\n            const url = new URL(await google.colab.kernel.proxyPort(%PORT%, {'cache': true}));\n            url.searchParams.set('tensorboardColab', 'true');\n            const iframe = document.createElement('iframe');\n            iframe.src = url;\n            iframe.setAttribute('width', '100%');\n            iframe.setAttribute('height', '%HEIGHT%');\n            iframe.setAttribute('frameborder', 0);\n            document.body.appendChild(iframe);\n        })();\n    "
    replacements = [('%PORT%', '%d' % port), ('%HEIGHT%', '%d' % height)]
    for k, v in replacements:
        shell = shell.replace(k, v)
    script = IPython.display.Javascript(shell)
    if display_handle:
        display_handle.update(script)
    else:
        IPython.display.display(script)