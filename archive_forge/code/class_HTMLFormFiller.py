import re
import six
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT
class HTMLFormFiller(object):
    """A stream filter that can populate HTML forms from a dictionary of values.
    
    >>> from genshi.input import HTML
    >>> html = HTML('''<form>
    ...   <p><input type="text" name="foo" /></p>
    ... </form>''', encoding='utf-8')
    >>> filler = HTMLFormFiller(data={'foo': 'bar'})
    >>> print(html | filler)
    <form>
      <p><input type="text" name="foo" value="bar"/></p>
    </form>
    """

    def __init__(self, name=None, id=None, data=None, passwords=False):
        """Create the filter.
        
        :param name: The name of the form that should be populated. If this
                     parameter is given, only forms where the ``name`` attribute
                     value matches the parameter are processed.
        :param id: The ID of the form that should be populated. If this
                   parameter is given, only forms where the ``id`` attribute
                   value matches the parameter are processed.
        :param data: The dictionary of form values, where the keys are the names
                     of the form fields, and the values are the values to fill
                     in.
        :param passwords: Whether password input fields should be populated.
                          This is off by default for security reasons (for
                          example, a password may end up in the browser cache)
        :note: Changed in 0.5.2: added the `passwords` option
        """
        self.name = name
        self.id = id
        if data is None:
            data = {}
        self.data = data
        self.passwords = passwords

    def __call__(self, stream):
        """Apply the filter to the given stream.
        
        :param stream: the markup event stream to filter
        """
        in_form = in_select = in_option = in_textarea = False
        select_value = option_value = textarea_value = None
        option_start = None
        option_text = []
        no_option_value = False
        for kind, data, pos in stream:
            if kind is START:
                tag, attrs = data
                tagname = tag.localname
                if tagname == 'form' and (self.name and attrs.get('name') == self.name or (self.id and attrs.get('id') == self.id) or (not (self.id or self.name))):
                    in_form = True
                elif in_form:
                    if tagname == 'input':
                        type = attrs.get('type', '').lower()
                        if type in ('checkbox', 'radio'):
                            name = attrs.get('name')
                            if name and name in self.data:
                                value = self.data[name]
                                declval = attrs.get('value')
                                checked = False
                                if isinstance(value, (list, tuple)):
                                    if declval is not None:
                                        u_vals = [six.text_type(v) for v in value]
                                        checked = declval in u_vals
                                    else:
                                        checked = any(value)
                                elif declval is not None:
                                    checked = declval == six.text_type(value)
                                elif type == 'checkbox':
                                    checked = bool(value)
                                if checked:
                                    attrs |= [(QName('checked'), 'checked')]
                                elif 'checked' in attrs:
                                    attrs -= 'checked'
                        elif type in ('', 'hidden', 'text') or (type == 'password' and self.passwords):
                            name = attrs.get('name')
                            if name and name in self.data:
                                value = self.data[name]
                                if isinstance(value, (list, tuple)):
                                    value = value[0]
                                if value is not None:
                                    attrs |= [(QName('value'), six.text_type(value))]
                    elif tagname == 'select':
                        name = attrs.get('name')
                        if name in self.data:
                            select_value = self.data[name]
                            in_select = True
                    elif tagname == 'textarea':
                        name = attrs.get('name')
                        if name in self.data:
                            textarea_value = self.data.get(name)
                            if isinstance(textarea_value, (list, tuple)):
                                textarea_value = textarea_value[0]
                            in_textarea = True
                    elif in_select and tagname == 'option':
                        option_start = (kind, data, pos)
                        option_value = attrs.get('value')
                        if option_value is None:
                            no_option_value = True
                            option_value = ''
                        in_option = True
                        continue
                yield (kind, (tag, attrs), pos)
            elif in_form and kind is TEXT:
                if in_select and in_option:
                    if no_option_value:
                        option_value += data
                    option_text.append((kind, data, pos))
                    continue
                elif in_textarea:
                    continue
                yield (kind, data, pos)
            elif in_form and kind is END:
                tagname = data.localname
                if tagname == 'form':
                    in_form = False
                elif tagname == 'select':
                    in_select = False
                    select_value = None
                elif in_select and tagname == 'option':
                    if isinstance(select_value, (tuple, list)):
                        selected = option_value in [six.text_type(v) for v in select_value]
                    else:
                        selected = option_value == six.text_type(select_value)
                    okind, (tag, attrs), opos = option_start
                    if selected:
                        attrs |= [(QName('selected'), 'selected')]
                    elif 'selected' in attrs:
                        attrs -= 'selected'
                    yield (okind, (tag, attrs), opos)
                    if option_text:
                        for event in option_text:
                            yield event
                    in_option = False
                    no_option_value = False
                    option_start = option_value = None
                    option_text = []
                elif in_textarea and tagname == 'textarea':
                    if textarea_value:
                        yield (TEXT, six.text_type(textarea_value), pos)
                        textarea_value = None
                    in_textarea = False
                yield (kind, data, pos)
            else:
                yield (kind, data, pos)