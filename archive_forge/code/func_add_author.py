from PySide2.QtSql import QSqlDatabase, QSqlError, QSqlQuery
from datetime import date
def add_author(q, name, birthdate):
    q.addBindValue(name)
    q.addBindValue(str(birthdate))
    q.exec_()
    return q.lastInsertId()