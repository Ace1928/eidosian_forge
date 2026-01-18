import dropbox

# Dropbox access token
ACCESS_TOKEN = "your_access_token"

# Files to backup
files_to_backup = ["file1.txt", "file2.txt", "file3.txt"]

# Create a Dropbox client
dbx = dropbox.Dropbox(ACCESS_TOKEN)

# Backup each file to Dropbox
for file in files_to_backup:
    with open(file, "rb") as f:
        dbx.files_upload(f.read(), f"/backup/{file}")
        print(f"Backed up: {file}")

print("File backup completed.")
