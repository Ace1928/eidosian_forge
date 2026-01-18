import logging
from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel
class EchoKernelApp(IPKernelApp):
    kernel_class = EchoKernel