from ..config import config
def _add_css_classes(item, css_classes):
    if not item.css_classes:
        item.css_classes = css_classes
    else:
        new_classes = [css_class for css_class in css_classes if css_class not in item.css_classes]
        item.css_classes = item.css_classes + new_classes