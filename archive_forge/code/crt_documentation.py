import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
Create a CRTTransferManager for optimized data transfer.