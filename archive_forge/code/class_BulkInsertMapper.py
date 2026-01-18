import datetime
import decimal
from .base import Database
class BulkInsertMapper:
    BLOB = 'TO_BLOB(%s)'
    DATE = 'TO_DATE(%s)'
    INTERVAL = 'CAST(%s as INTERVAL DAY(9) TO SECOND(6))'
    NCLOB = 'TO_NCLOB(%s)'
    NUMBER = 'TO_NUMBER(%s)'
    TIMESTAMP = 'TO_TIMESTAMP(%s)'
    types = {'AutoField': NUMBER, 'BigAutoField': NUMBER, 'BigIntegerField': NUMBER, 'BinaryField': BLOB, 'BooleanField': NUMBER, 'DateField': DATE, 'DateTimeField': TIMESTAMP, 'DecimalField': NUMBER, 'DurationField': INTERVAL, 'FloatField': NUMBER, 'IntegerField': NUMBER, 'PositiveBigIntegerField': NUMBER, 'PositiveIntegerField': NUMBER, 'PositiveSmallIntegerField': NUMBER, 'SmallAutoField': NUMBER, 'SmallIntegerField': NUMBER, 'TextField': NCLOB, 'TimeField': TIMESTAMP}