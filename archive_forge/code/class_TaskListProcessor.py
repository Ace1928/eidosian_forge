import markdown
from functools import reduce
from markdown.treeprocessors import Treeprocessor
import xml.etree.ElementTree as etree
class TaskListProcessor(Treeprocessor):

    def __init__(self, ext):
        super(TaskListProcessor, self).__init__()
        self.ext = ext

    def run(self, root):
        ordered = self.ext.getConfig('ordered')
        unordered = self.ext.getConfig('unordered')
        if not ordered and (not unordered):
            return root
        checked_pattern = _to_list(self.ext.getConfig('checked'))
        unchecked_pattern = _to_list(self.ext.getConfig('unchecked'))
        if not checked_pattern and (not unchecked_pattern):
            return root
        prefix_length = reduce(max, (len(e) for e in checked_pattern + unchecked_pattern))
        item_attrs = self.ext.getConfig('item_attrs')
        list_attrs = self.ext.getConfig('list_attrs')
        base_cb_attrs = self.ext.getConfig('checkbox_attrs')
        max_depth = self.ext.getConfig('max_depth')
        if not max_depth:
            max_depth = float('inf')
        lists = set()
        stack = [(root, None, 0)]
        while stack:
            el, parent, depth = stack.pop()
            if parent and el.tag == 'li' and (parent.tag == 'ul' and unordered or (parent.tag == 'ol' and ordered)):
                depth += 1
                text = (el.text or '').lstrip()
                lower_text = text[:prefix_length].lower()
                found = False
                checked = False
                for p in checked_pattern:
                    if lower_text.startswith(p):
                        found = True
                        checked = True
                        text = text[len(p):]
                        break
                else:
                    for p in unchecked_pattern:
                        if lower_text.startswith(p):
                            found = True
                            text = text[len(p):]
                            break
                if found:
                    if list_attrs:
                        lists.add((parent, depth))
                    attrs = {'type': 'checkbox', 'disabled': 'disabled'}
                    if checked:
                        attrs['checked'] = 'checked'
                    attrs.update(base_cb_attrs(parent, el) if callable(base_cb_attrs) else base_cb_attrs)
                    checkbox = etree.Element('input', attrs)
                    checkbox.tail = text
                    el.text = ''
                    el.insert(0, checkbox)
                    for k, v in (item_attrs(parent, el, checkbox) if callable(item_attrs) else item_attrs).items():
                        el.set(k, v)
            if depth < max_depth:
                for child in el:
                    stack.append((child, el, depth))
        for list, depth in lists:
            for k, v in (list_attrs(list, depth) if callable(list_attrs) else list_attrs).items():
                list.set(k, v)
        return root