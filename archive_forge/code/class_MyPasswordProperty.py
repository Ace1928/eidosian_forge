import unittest
import logging
import time
class MyPasswordProperty(PasswordProperty):
    data_type = MyPassword
    type_name = MyPassword.__name__