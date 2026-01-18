from ..config import config
def _remove_css_classes(item, css_classes):
    if not item.css_classes:
        return
    item.css_classes = [css_class for css_class in item.css_classes if css_class not in css_classes]