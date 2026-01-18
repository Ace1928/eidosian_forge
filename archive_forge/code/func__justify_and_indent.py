import zope.interface
def _justify_and_indent(text, level, munge=0, width=72):
    """ indent and justify text, rejustify (munge) if specified """
    indent = ' ' * level
    if munge:
        lines = []
        line = indent
        text = text.split()
        for word in text:
            line = ' '.join([line, word])
            if len(line) > width:
                lines.append(line)
                line = indent
        else:
            lines.append(line)
        return '\n'.join(lines)
    else:
        return indent + text.strip().replace('\r\n', '\n').replace('\n', '\n' + indent)