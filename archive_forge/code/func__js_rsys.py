import json
from .web import CSSPrinter, _html_clsname
def _js_rsys(cls_names_substances, substance_row_cls_irrel, rsys_id):
    if cls_names_substances is None:
        return ''
    return _js_rsys_template % dict(cls_names_substances=json.dumps(cls_names_substances), substance_row_cls_irrel=json.dumps(substance_row_cls_irrel), rsys_id=rsys_id)