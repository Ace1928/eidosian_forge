from gettext import NullTranslations
import os
import re
from functools import partial
from types import FunctionType
import six
from genshi.core import Attrs, Namespace, QName, START, END, TEXT, \
from genshi.template.base import DirectiveFactory, EXPR, SUB, _apply_directives
from genshi.template.directives import Directive, StripDirective
from genshi.template.markup import MarkupTemplate, EXEC
from genshi.compat import ast, IS_PYTHON2, _ast_Str, _ast_Str_value
class MessageBuffer(object):
    """Helper class for managing internationalized mixed content.

    :since: version 0.5
    """

    def __init__(self, directive=None):
        """Initialize the message buffer.

        :param directive: the directive owning the buffer
        :type directive: I18NDirective
        """
        self.orig_params = self.params = directive.params[:]
        self.directive = directive
        self.string = []
        self.events = {}
        self.values = {}
        self.depth = 1
        self.order = 1
        self._prev_order = None
        self.stack = [0]
        self.subdirectives = {}

    def _add_event(self, order, event):
        if order == self._prev_order:
            self.events[order][-1].append(event)
        else:
            self._prev_order = order
            self.events.setdefault(order, [])
            self.events[order].append([event])

    def append(self, kind, data, pos):
        """Append a stream event to the buffer.

        :param kind: the stream event kind
        :param data: the event data
        :param pos: the position of the event in the source
        """
        if kind is SUB:
            order = self.stack[-1] + 1
            subdirectives, substream = data
            self.subdirectives.setdefault(order, []).extend(subdirectives)
            self._add_event(order, (SUB_START, None, pos))
            for skind, sdata, spos in substream:
                self.append(skind, sdata, spos)
            self._add_event(order, (SUB_END, None, pos))
        elif kind is TEXT:
            if '[' in data or ']' in data:
                data = data.replace('[', '\\[').replace(']', '\\]')
            self.string.append(data)
            self._add_event(self.stack[-1], (kind, data, pos))
        elif kind is EXPR:
            if self.params:
                param = self.params.pop(0)
            else:
                params = ', '.join(['"%s"' % p for p in self.orig_params if p])
                if params:
                    params = '(%s)' % params
                raise IndexError("%d parameters%s given to 'i18n:%s' but %d or more expressions used in '%s', line %s" % (len(self.orig_params), params, self.directive.tagname, len(self.orig_params) + 1, os.path.basename(pos[0] or 'In-memory Template'), pos[1]))
            self.string.append('%%(%s)s' % param)
            self._add_event(self.stack[-1], (kind, data, pos))
            self.values[param] = (kind, data, pos)
        elif kind is START:
            self.string.append('[%d:' % self.order)
            self.stack.append(self.order)
            self._add_event(self.stack[-1], (kind, data, pos))
            self.depth += 1
            self.order += 1
        elif kind is END:
            self.depth -= 1
            if self.depth:
                self._add_event(self.stack[-1], (kind, data, pos))
                self.string.append(']')
                self.stack.pop()

    def format(self):
        """Return a message identifier representing the content in the
        buffer.
        """
        return ''.join(self.string).strip()

    def translate(self, string, regex=re.compile('%\\((\\w+)\\)s')):
        """Interpolate the given message translation with the events in the
        buffer and return the translated stream.

        :param string: the translated message string
        """
        substream = None

        def yield_parts(string):
            for idx, part in enumerate(regex.split(string)):
                if idx % 2:
                    yield self.values[part]
                elif part:
                    yield (TEXT, part.replace('\\[', '[').replace('\\]', ']'), (None, -1, -1))
        parts = parse_msg(string)
        parts_counter = {}
        for order, string in parts:
            parts_counter.setdefault(order, []).append(None)
        while parts:
            order, string = parts.pop(0)
            events = self.events[order]
            if events:
                events = events.pop(0)
            else:
                events = [(TEXT, '', (None, -1, -1))]
            parts_counter[order].pop()
            for event in events:
                if event[0] is SUB_START:
                    substream = []
                elif event[0] is SUB_END:
                    yield (SUB, (self.subdirectives[order], substream), event[2])
                    substream = None
                elif event[0] is TEXT:
                    if string:
                        for part in yield_parts(string):
                            if substream is not None:
                                substream.append(part)
                            else:
                                yield part
                        string = None
                elif event[0] is START:
                    if substream is not None:
                        substream.append(event)
                    else:
                        yield event
                    if string:
                        for part in yield_parts(string):
                            if substream is not None:
                                substream.append(part)
                            else:
                                yield part
                        string = None
                elif event[0] is END:
                    if string:
                        for part in yield_parts(string):
                            if substream is not None:
                                substream.append(part)
                            else:
                                yield part
                        string = None
                    if substream is not None:
                        substream.append(event)
                    else:
                        yield event
                elif event[0] is EXPR:
                    continue
                else:
                    if string:
                        for part in yield_parts(string):
                            if substream is not None:
                                substream.append(part)
                            else:
                                yield part
                        string = None
                    if substream is not None:
                        substream.append(event)
                    else:
                        yield event