from django.db.models import DateTimeField, Func, UUIDField
class TransactionNow(Func):
    template = 'CURRENT_TIMESTAMP'
    output_field = DateTimeField()