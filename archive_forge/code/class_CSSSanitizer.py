import tinycss2
class CSSSanitizer:

    def __init__(self, allowed_css_properties=ALLOWED_CSS_PROPERTIES, allowed_svg_properties=ALLOWED_SVG_PROPERTIES):
        self.allowed_css_properties = allowed_css_properties
        self.allowed_svg_properties = allowed_svg_properties

    def sanitize_css(self, style):
        """Sanitizes css in style tags"""
        parsed = tinycss2.parse_declaration_list(style)
        if not parsed:
            return ''
        new_tokens = []
        for token in parsed:
            if token.type == 'declaration':
                if token.lower_name in self.allowed_css_properties or token.lower_name in self.allowed_svg_properties:
                    new_tokens.append(token)
            elif token.type in ('comment', 'whitespace'):
                if new_tokens and new_tokens[-1].type != token.type:
                    new_tokens.append(token)
        if not new_tokens:
            return ''
        return tinycss2.serialize(new_tokens).strip()