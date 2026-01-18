from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import time
from paste.exceptions import formatter
class FileReporter(Reporter):
    file = None
    show_hidden_frames = True

    def check_params(self):
        assert self.file is not None, 'You must give a file object'

    def report(self, exc_data):
        text = self.format_text(exc_data, show_hidden_frames=self.show_hidden_frames)
        self.file.write(text + '\n' + '-' * 60 + '\n')