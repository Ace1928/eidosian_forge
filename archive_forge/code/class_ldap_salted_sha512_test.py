from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class ldap_salted_sha512_test(HandlerCase):
    handler = hash.ldap_salted_sha512
    known_correct_hashes = [('toomanysecrets', '{SSHA512}wExp4xjiCHS0zidJDC4UJq9EEeIebAQPJ1PWSwfhxWjfutI9XiiKuHm2AE41cEFfK+8HyI8bh+ztbczUGsvVFIgICWWPt7qu'), (u('letmèïn'), '{SSHA512}mpNUSmZc3TNx+RnPwkIAVMf7ocEKLPrIoQNsg4Eu8dHvyCeb2xzHp5A6n4tF7ntknSvfvRZaJII4ImvNJlYsgiwAm0FMqR+3'), ('password', '{SSHA512}f/lFQskkl7PdMsTGJxHZq8LDt/l+UqRMm6/pj4pV7/xZkcOaKCgvQqp+KCeXc/Vd4RY6vEHWn4y0DnFcQ6wgyv9fyxk='), ('test', '{SSHA512}Tgx/uhHnlM9/GgQvI31dN7cheDXg7WypZwaaIkyRsgV/BKIzBG3G/wUd9o1dpi06p3SYzMedg0lvTc3b6CtdO0Xo/f9/L+Uc'), ('test', '{SSHA512}Yg9DQ2wURCFGwobu7R2O6cq7nVbnGMPrFCX0aPQ9kj/y1hd6k9PEzkgWCB5aXdPwPzNrVb0PkiHiBnG1CxFiT+B8L8U='), ('test', '{SSHA512}5ecDGWs5RY4xLszUO6hAcl90W3wAozGQoI4Gqj8xSZdcfU1lVEM4aY8s+4xVeLitcn7BO8i7xkzMFWLoxas7SeHc23sP4dx77937PyeE0A=='), ('test', '{SSHA512}6FQv5W47HGg2MFBFZofoiIbO8KRW75Pm51NKoInpthYQQ5ujazHGhVGzrj3JXgA7j0k+UNmkHdbJjdY5xcUHPzynFEII4fwfIySEcG5NKSU=')]
    known_malformed_hashes = ['{SSHA512}zFnn4/8x8GveUaMqgrYWyIWqFQ0Irt6gADPtRk4Uv3nUC6uR5cD8+YdQni/0ZNij9etm6p17kSFuww3M6l+d6AbAeA==', '{SSHA512}Tgx/uhHnlM9/GgQvI31dN7cheDXg7WypZwaaIkyRsgV/BKIzBG3G/wUd9o1dpi06p3SYzMedg0lvTc3b6CtdO0Xo/f9/L+U', '{SSHA512}Tgx/uhHnlM9/GgQvI31dN7cheDXg7WypZwaaIkyRsgV/BKIzBG3G/wUd9o1dpi06p3SYzMedg0lvTc3b6CtdO0Xo/f9/L+U@', '{SSHA512}Tgx/uhHnlM9/GgQvI31dN7cheDXg7WypZwaaIkyRsgV/BKIzBG3G/wUd9o1dpi06p3SYzMedg0lvTc3b6CtdO0Xo/f9/L+U===']