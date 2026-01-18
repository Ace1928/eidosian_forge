import re
def _rewrite_list_item(tok):
    children = tok['children']
    if children:
        first_child = children[0]
        text = first_child.get('text', '')
        m = TASK_LIST_ITEM.match(text)
        if m:
            mark = m.group(1)
            first_child['text'] = text[m.end():]
            tok['type'] = 'task_list_item'
            tok['attrs'] = {'checked': mark != '[ ]'}