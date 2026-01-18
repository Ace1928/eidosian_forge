from itertools import chain
from genshi.core import Attrs, Markup, Namespace, Stream
from genshi.core import START, END, START_NS, END_NS, TEXT, PI, COMMENT
from genshi.input import XMLParser
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.interpolation import interpolate
from genshi.template.directives import *
from genshi.template.text import NewTextTemplate
def _match(self, stream, ctxt, start=0, end=None, **vars):
    """Internal stream filter that applies any defined match templates
        to the stream.
        """
    match_templates = ctxt._match_templates

    def _strip(stream, append):
        depth = 1
        while 1:
            event = next(stream)
            if event[0] is START:
                depth += 1
            elif event[0] is END:
                depth -= 1
            if depth > 0:
                yield event
            else:
                append(event)
                break
    for event in stream:
        if not match_templates or (event[0] is not START and event[0] is not END):
            yield event
            continue
        for idx, (test, path, template, hints, namespaces, directives) in enumerate(match_templates):
            if idx < start or (end is not None and idx >= end):
                continue
            if test(event, namespaces, ctxt) is True:
                if 'match_once' in hints:
                    del match_templates[idx]
                    idx -= 1
                for test in [mt[0] for mt in match_templates[idx + 1:]]:
                    test(event, namespaces, ctxt, updateonly=True)
                pre_end = idx + 1
                if 'match_once' not in hints and 'not_recursive' in hints:
                    pre_end -= 1
                tail = []
                inner = _strip(stream, tail.append)
                if pre_end > 0:
                    inner = self._match(inner, ctxt, start=start, end=pre_end, **vars)
                content = self._include(chain([event], inner, tail), ctxt)
                if 'not_buffered' not in hints:
                    content = list(content)
                content = Stream(content)
                selected = [False]

                def select(path):
                    selected[0] = True
                    return content.select(path, namespaces, ctxt)
                vars = dict(select=select)
                template = _apply_directives(template, directives, ctxt, vars)
                for event in self._match(self._flatten(template, ctxt, **vars), ctxt, start=idx + 1, **vars):
                    yield event
                if not selected[0]:
                    for event in content:
                        pass
                for test in [mt[0] for mt in match_templates[idx:]]:
                    test(tail[0], namespaces, ctxt, updateonly=True)
                break
        else:
            yield event