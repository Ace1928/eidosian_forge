from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import time
from paste.exceptions import formatter
class LogReporter(Reporter):
    filename = None
    show_hidden_frames = True

    def check_params(self):
        assert self.filename is not None, 'You must give a filename'

    def report(self, exc_data):
        text = self.format_text(exc_data, show_hidden_frames=self.show_hidden_frames)
        f = open(self.filename, 'a')
        try:
            f.write(text + '\n' + '-' * 60 + '\n')
        finally:
            f.close()