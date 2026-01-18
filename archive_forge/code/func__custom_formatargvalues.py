import asyncio
import io
import inspect
import logging
import os
import queue
import uuid
import sys
import threading
import time
from typing_extensions import Literal
from werkzeug.serving import make_server
def _custom_formatargvalues(args, varargs, varkw, locals, formatarg=str, formatvarargs=lambda name: '*' + name, formatvarkw=lambda name: '**' + name, formatvalue=lambda value: '=' + repr(value)):
    """Copied from inspect.formatargvalues, modified to place function
    arguments on separate lines"""

    def convert(name, locals=locals, formatarg=formatarg, formatvalue=formatvalue):
        return formatarg(name) + formatvalue(locals[name])
    specs = []
    for i in range(len(args)):
        specs.append(convert(args[i]))
    if varargs:
        specs.append(formatvarargs(varargs) + formatvalue(locals[varargs]))
    if varkw:
        specs.append(formatvarkw(varkw) + formatvalue(locals[varkw]))
    result = '(' + ', '.join(specs) + ')'
    if len(result) < 40:
        return result
    return '(\n    ' + ',\n    '.join(specs) + '\n)'