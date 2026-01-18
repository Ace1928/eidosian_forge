from logging import handlers
from oslo_i18n import _translate
def _translate_and_log_record(self, record):
    record.msg = _translate.translate(record.msg, self.locale)
    record.args = _translate.translate_args(record.args, self.locale)
    self.target.emit(record)