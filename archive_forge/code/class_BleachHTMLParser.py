import re
import string
import warnings
from bleach._vendor.html5lib import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib import (
from bleach._vendor.html5lib.constants import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib.constants import (
from bleach._vendor.html5lib.filters.base import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib._inputstream import (
from bleach._vendor.html5lib.serializer import (
from bleach._vendor.html5lib._tokenizer import (
from bleach._vendor.html5lib._trie import (
class BleachHTMLParser(HTMLParser):
    """Parser that uses BleachHTMLTokenizer"""

    def __init__(self, tags, strip, consume_entities, **kwargs):
        """
        :arg tags: set of allowed tags--everything else is either stripped or
            escaped; if None, then this doesn't look at tags at all
        :arg strip: whether to strip disallowed tags (True) or escape them (False);
            if tags=None, then this doesn't have any effect
        :arg consume_entities: whether to consume entities (default behavior) or
            leave them as is when tokenizing (BleachHTMLTokenizer-added behavior)

        """
        self.tags = frozenset((tag.lower() for tag in tags)) if tags is not None else None
        self.strip = strip
        self.consume_entities = consume_entities
        super().__init__(**kwargs)

    def _parse(self, stream, innerHTML=False, container='div', scripting=True, **kwargs):
        self.innerHTMLMode = innerHTML
        self.container = container
        self.scripting = scripting
        self.tokenizer = BleachHTMLTokenizer(stream=stream, consume_entities=self.consume_entities, parser=self, **kwargs)
        self.reset()
        try:
            self.mainLoop()
        except ReparseException:
            self.reset()
            self.mainLoop()