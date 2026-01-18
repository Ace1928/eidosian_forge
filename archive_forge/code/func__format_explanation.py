import sys
import py
from py._code._assertionnew import interpret as reinterpret
def _format_explanation(explanation):
    """This formats an explanation

    Normally all embedded newlines are escaped, however there are
    three exceptions: 
{, 
} and 
~.  The first two are intended
    cover nested explanations, see function and attribute explanations
    for examples (.visit_Call(), visit_Attribute()).  The last one is
    for when one explanation needs to span multiple lines, e.g. when
    displaying diffs.
    """
    raw_lines = (explanation or '').split('\n')
    lines = [raw_lines[0]]
    for l in raw_lines[1:]:
        if l.startswith('{') or l.startswith('}') or l.startswith('~'):
            lines.append(l)
        else:
            lines[-1] += '\\n' + l
    result = lines[:1]
    stack = [0]
    stackcnt = [0]
    for line in lines[1:]:
        if line.startswith('{'):
            if stackcnt[-1]:
                s = 'and   '
            else:
                s = 'where '
            stack.append(len(result))
            stackcnt[-1] += 1
            stackcnt.append(0)
            result.append(' +' + '  ' * (len(stack) - 1) + s + line[1:])
        elif line.startswith('}'):
            assert line.startswith('}')
            stack.pop()
            stackcnt.pop()
            result[stack[-1]] += line[1:]
        else:
            assert line.startswith('~')
            result.append('  ' * len(stack) + line[1:])
    assert len(stack) == 1
    return '\n'.join(result)