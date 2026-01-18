from __future__ import annotations
import re
from kombu.utils.text import escape_regex
class TopicExchange(ExchangeType):
    """Topic exchange.

    The `topic` exchange routes messages based on words separated by
    dots, using wildcard characters ``*`` (any single word), and ``#``
    (one or more words).
    """
    type = 'topic'
    wildcards = {'*': '.*?[^\\.]', '#': '.*?'}
    _compiled = {}

    def lookup(self, table, exchange, routing_key, default):
        return {queue for rkey, pattern, queue in table if self._match(pattern, routing_key)}

    def deliver(self, message, exchange, routing_key, **kwargs):
        _lookup = self.channel._lookup
        _put = self.channel._put
        deadletter = self.channel.deadletter_queue
        for queue in [q for q in _lookup(exchange, routing_key) if q and q != deadletter]:
            _put(queue, message, **kwargs)

    def prepare_bind(self, queue, exchange, routing_key, arguments):
        return (routing_key, self.key_to_pattern(routing_key), queue)

    def key_to_pattern(self, rkey):
        """Get the corresponding regex for any routing key."""
        return '^%s$' % '\\.'.join((self.wildcards.get(word, word) for word in escape_regex(rkey, '.#*').split('.')))

    def _match(self, pattern, string):
        """Match regular expression (cached).

        Same as :func:`re.match`, except the regex is compiled and cached,
        then reused on subsequent matches with the same pattern.
        """
        try:
            compiled = self._compiled[pattern]
        except KeyError:
            compiled = self._compiled[pattern] = re.compile(pattern, re.U)
        return compiled.match(string)