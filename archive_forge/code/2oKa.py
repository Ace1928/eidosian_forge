import os
import paramiko

# SSH connection details
hostname = "your_remote_server"
username = "your_username"
password = "your_password"

# Local directory and files to backup
local_dir = "/path/to/local/directory"
files_to_backup = ["file1.txt", "file2.txt", "file3.txt"]

# Remote directory for backups
remote_dir = "/path/to/remote/backup/directory"

# Create an SSH client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, username=username, password=password)

# Create an SFTP client
sftp = ssh.open_sftp()

# Backup each file
for file_name in files_to_backup:
    local_path = os.path.join(local_dir, file_name)
    remote_path = os.path.join(remote_dir, file_name)
    sftp.put(local_path, remote_path)

# Close the SFTP and SSH connections
sftp.close()
ssh.close()
