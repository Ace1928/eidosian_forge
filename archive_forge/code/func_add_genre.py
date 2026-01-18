from PySide2.QtSql import QSqlDatabase, QSqlError, QSqlQuery
from datetime import date
def add_genre(q, name):
    q.addBindValue(name)
    q.exec_()
    return q.lastInsertId()